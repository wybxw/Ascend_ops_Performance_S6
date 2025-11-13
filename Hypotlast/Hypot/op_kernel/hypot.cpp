// ...existing code...
#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;

/*
  Hypot operator: out = sqrt(x*x + y*y)
  支持类型：float / half / bfloat16
  规则：
    - T == float : 直接用 float 计算（无需额外缓冲）
    - T == half  : 直接用 half 计算（无需额外缓冲）
    - T == bfloat16_t : 提升到 float 计算（需要 Cast，使用额外 VECCALC 缓冲）
*/
static constexpr int BUFFER_NUM = 2;
template <typename T>
struct IsHypotSupported : std::false_type {};
template <> struct IsHypotSupported<float>       : std::true_type {};
template <> struct IsHypotSupported<half>        : std::true_type {};
template <> struct IsHypotSupported<bfloat16_t>  : std::true_type {};

template <typename T>
class KernelHypot {
public:
    static_assert(IsHypotSupported<T>::value, "KernelHypot: only float/half/bfloat16 supported");

    

    // ComputeT: bfloat16 -> float , half->float其它类型保持不变
    using ComputeT = std::conditional_t<std::is_same_v<T, bfloat16_t>, float, T>;
    // using ComputeT = float;
    static constexpr bool NEED_CAST = !std::is_same_v<ComputeT, T>;
    
    
    int32_t BLOCKLENGTH = 61440/sizeof(ComputeT);
    __aicore__ inline KernelHypot() = default;

    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR y_gm, GM_ADDR z_gm,
                                const TilingData &tiling, TPipe *pipe_ptr)
    {
        blockIdx  = GetBlockIdx();
        blockNum  = GetBlockNum();
        totalSize = tiling.totalLength;
        pipe = pipe_ptr;
        if(totalSize <= blockNum * BLOCKLENGTH){
            BLOCKLENGTH = (totalSize+blockNum-1)/blockNum;
            BLOCKLENGTH = (BLOCKLENGTH + 32 -1 )& ~(31);
        }
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x_gm), totalSize);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y_gm), totalSize);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(z_gm), totalSize);
        // AscendC::DataCachePreload(xGm,0);
        // AscendC::DataCachePreload(yGm,0);
        // xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // zGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        
        // VECIN/VECOUT 用原类型 T（用于搬入/搬出）
        pipe->InitBuffer(inBufX,  BUFFER_NUM, BLOCKLENGTH * sizeof(T) + 256);
        pipe->InitBuffer(inBufY,  BUFFER_NUM, BLOCKLENGTH * sizeof(T) + 256);
        // pipe->InitBuffer(outBufT, BUFFER_NUM, BLOCKLENGTH * sizeof(T) + 256);
        pipe->InitBuffer(calcBuf,BLOCKLENGTH * sizeof(ComputeT) + 256);

        // 仅在需要提升时，初始化 VECCALC 计算缓冲
        if constexpr (NEED_CAST) {
            pipe->InitBuffer(InCastX,BLOCKLENGTH * sizeof(ComputeT) + 256);
            // pipe->InitBuffer(InCastY,BLOCKLENGTH * sizeof(ComputeT) + 256);
        }
    }

    __aicore__ inline void Process()
    {
        // if(blockIdx==0)
        // for (int64_t t = blockIdx;; t++) {
        for (int64_t t = blockIdx;; t+=blockNum) {
            int64_t base = t * BLOCKLENGTH;
            int64_t remain = (base + BLOCKLENGTH <= totalSize) ? BLOCKLENGTH : (totalSize > base ? totalSize - base : 0u);
            if(remain<=0) break;
            CopyIn(base, remain);
            Compute(remain);
            CopyOut(base, remain);
            // if (base>=totalSize) break;
        }
    }
    __aicore__ inline void CopyIn(int64_t base, int64_t len)
    {
        // 统一以原类型 T 搬入到 VECIN 队列
        auto tx = inBufX.AllocTensor<T>();
        auto ty = inBufY.AllocTensor<T>();
        DataCopy(tx, xGm[base], len);
        DataCopy(ty, yGm[base], len);
        inBufX.EnQue(tx);
        inBufY.EnQue(ty);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        if constexpr (!NEED_CAST) {
            // assert(0);
            // 直接路径：T == ComputeT（float/half）
            // 直接从 VECIN 读取 T，立即在该缓冲上进行计算（把 VECIN 当作计算缓冲）
            auto x = inBufX.DeQue<T>();
            auto y = inBufY.DeQue<T>();
            auto out = y; // 直接写回原类型缓冲
            auto tmp = calcBuf.Get<ComputeT>(); // 临时用于 y*y

            Mul(out, x, x, len);   // out = x*x//文档说是不能src重叠,实际可以,不知道为啥
            MulAddDst(out, y, y, len);
            inBufX.FreeTensor(x);
            Sqrt(out, out, len);
            // 释放输入与临时
            inBufY.EnQue(out);
        } else {
            // Cast 路径：T (bfloat16) -> ComputeT (float) -> compute -> back to T
            // 1) 从 VECIN 读原始 bfloat16 数据
            auto txT = inBufX.DeQue<T>();
            auto tyT = inBufY.DeQue<T>();
            auto out = tyT;
            // 2) 为计算分配 VECCALC 型缓冲并 Cast 到 ComputeT
            auto cx = InCastX.Get<ComputeT>();
            auto tmp =calcBuf.Get<ComputeT>();

            
            Cast(cx, txT, AscendC::RoundMode::CAST_NONE, len);
            Mul(tmp, cx, cx, len);
            inBufX.FreeTensor(txT);
            Cast(cx, tyT, AscendC::RoundMode::CAST_NONE, len);
            MulAddDst(tmp, cx, cx, len);
            Sqrt(tmp, tmp, len);

            Cast(out,tmp,AscendC::RoundMode::CAST_RINT,len);


            

            // 5) 将计算结果入队等待搬出（结果类型为 ComputeT）
            inBufY.EnQue(out);
            
        }
    }

    __aicore__ inline void CopyOut(uint32_t base, uint32_t len)
    {
            // 直接路径：outBufT 已经包含结果（T）
        auto outT = inBufY.DeQue<T>();
        DataCopy(zGm[base], outT,len);
        inBufY.FreeTensor(outT);
    }
private:
    TPipe *pipe = nullptr;
    
    // VECIN/VECOUT 始终为原始类型 T（专用于搬入/搬出）
    TQue<TPosition::VECIN, BUFFER_NUM>  inBufX;
    TQueBind<TPosition::VECIN,TPosition::VECOUT,BUFFER_NUM>  inBufY;

    // VECCALC 仅在需要提升时存在（ComputeT），否则不使用
    // 使用模板条件编译：声明但仅在 Init 中创建
    TBuf<TPosition::VECCALC> calcBuf,InCastX;
    // LocalTensor<T> outT
    AscendC::GlobalTensor<T> xGm, yGm, zGm;
    const int pad =(32/sizeof(T)) - 1;
    int64_t blockIdx = 0;
    int64_t blockNum = 1;
    int64_t tileNum  = 0;
    int64_t totalSize = 0;


};
template<std::size_t Bytes> struct int_of_bytes;
template<> struct int_of_bytes<1> { using type = std::int8_t;  };
template<> struct int_of_bytes<2> { using type = std::int16_t; };
template<> struct int_of_bytes<4> { using type = std::int32_t; };
template<> struct int_of_bytes<8> { using type = std::int64_t; };

template<std::size_t Bytes>
using int_of_bytes_t = typename int_of_bytes<Bytes>::type;


// UB 宏
static constexpr int32_t UB_BYTES = 189* 1024;       // 189KB
static constexpr int32_t UB_BUF_RESERVE = 0 * 1024; // 4KB 保留给控制结构等
static constexpr int32_t MIN_BLOCK_BYTES = 32;        // 32B 对齐最小块

__aicore__ inline int32_t gcd_32(int32_t a, int32_t b) {
    if (a == 0) return b;
    if (b == 0) return a;
    while (b) {
        uint64_t r = a % b;
        a = b; b = r;
    }
    return a;
}
template <typename T>
constexpr bool KEXP_IsAllowedType_v = std::disjunction_v<
    std::is_same<T, int8_t>,
    std::is_same<T, int16_t>,
    std::is_same<T, int32_t>,
    std::is_same<T, int64_t>
    // std::is_same<T, bfloat16_t>
    >;
template <typename T, typename = std::enable_if_t<KEXP_IsAllowedType_v<T>>>
class KernelExpand
{
    static_assert(KEXP_IsAllowedType_v<T>, "");

public:
    __aicore__ inline KernelExpand() {}
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, GM_ADDR workspace, TilingData &tiling, AscendC::TPipe *pipein,bool flag)
    {
        this->pipe = pipein;
        this->blockIdx = AscendC::GetBlockIdx();
        this->blockStride =AscendC::GetBlockNum();

        if(!flag) {
            this->total_outer_repeats =this->num_tile_entries = tiling.Expandsize_x1;
            for (int i = 0; i < 4; ++i)
            {
                this->outer[i] = tiling.outer_x1[i];
                this->repeater[i] = tiling.repeater_x1[i];
                this->inner[i] = tiling.inner_x1[i];
            }
        }else{
            this->total_outer_repeats =this->num_tile_entries = tiling.Expandsize_x2;
            for (int i = 0; i < 4; ++i)
            {
                this->outer[i] = tiling.outer_x2[i];
                this->repeater[i] = tiling.repeater_x2[i];
                this->inner[i] = tiling.inner_x2[i];
            }
        }
        // srcGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // dstGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // wsGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        
        this->outputsize = tiling.totalLength;
        this->ws_addr = workspace;
        this->src_addr = src;
        this->dst_addr = dst;
        this->dtype_bytes = sizeof(T);
        int32_t per_buf_bytes =  189 * 1024 / BUFFER_NUM;                
        this->ub_buf_elems = per_buf_bytes / this->dtype_bytes;
        this->align_elems = max((int32_t)1, (int32_t)(MIN_BLOCK_BYTES + this->dtype_bytes - 1) / this->dtype_bytes);
        pipe->InitBuffer(Queue, BUFFER_NUM, per_buf_bytes);
    }

    __aicore__ inline void Process()
    {
        for (int32_t idx = 0; idx < this->total_outer_repeats; ++idx)
        {

            if ((this->total_outer_repeats & 1) == ((idx + 1) & 1)) srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ws_addr)),dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst_addr));
            else srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst_addr)),dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ws_addr));
            if (idx == 0) srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src_addr));
            int32_t align_inner= (inner[idx]+align_elems-1)/align_elems*align_elems;
            int32_t step = min( ub_buf_elems / align_inner,outer[idx] / 40);
            if(this->outer[idx]>=this->blockStride){
                for (int32_t outer_idx = step*this->blockIdx; outer_idx < this->outer[idx]; outer_idx+=step*this->blockStride)
                {
                    performTileBroadcast(idx, outer_idx,min(step,outer[idx]-outer_idx));
                }
            }else{
                performTileBroadcast(idx, GetBlockIdx()%this->outer[idx]); //线程下沉给(1,repeat,inner)
            }
            SyncAll();
        }
        
    }

    // 计算基址（以元素计）
    __aicore__ inline int32_t compute_in_base(int tile_idx, int32_t outer_index)
    {
        return outer_index * (int32_t)this->inner[tile_idx];
    }
    __aicore__ inline int32_t compute_out_base(int tile_idx, int32_t outer_index)
    {
        return outer_index * (int32_t)(this->inner[tile_idx] * this->repeater[tile_idx]);
    }

    __aicore__ inline bool is32AlignedElem(int32_t offset_elems) const
    {
        return (offset_elems * this->dtype_bytes) % 32 == 0;
    }

    // 对齐拷贝
    __aicore__ inline void MyCopy(AscendC::LocalTensor<T> &dstBase,int32_t elems)
    {
        //假设已经对齐
        Adds<T>(dstBase[elems],dstBase,T(0),elems);
    }
    __aicore__ inline void MyFillPad(AscendC::LocalTensor<T> &dstBase,
                                    int32_t &filled)
    {
        //使用标量 将其复制到一个可接受的大小
        int repeat = this->align_elems/gcd_32(filled,this->align_elems);
        if(repeat*filled*sizeof(T) <= 4 * 1024){
            for(int i=1;i<repeat;i++)
                for(int j=0;j<filled;j++)
                    dstBase.SetValue(i*filled + j,dstBase.GetValue(j));
            filled *= repeat;
        }
    }
    __aicore__ inline void copyIn(
        int32_t in_base_elem,
        int32_t offset_elem,
        int32_t cur_elems)
    {
        auto buf = Queue.AllocTensor<T>();
        AscendC::DataCopy(buf, srcGm[in_base_elem + offset_elem], cur_elems + align_elems - 1);
        Queue.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN,T>(buf);
    }

    __aicore__ inline int32_t computeExpandVec(int32_t cur_elems,int32_t inner_elems,int32_t repeat)
    {

        int32_t filled = cur_elems ;
        LocalTensor<T> buf = Queue.DeQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN,T>();
        if (inner_elems == 1)
        {
            filled = min(this->ub_buf_elems, repeat); // 最多repeat成一行
            T value = buf.GetValue(0);
            if constexpr (std::is_same_v<T, int8_t>){
                // 按位解释为无符号
                Duplicate<int16_t>(buf.template ReinterpretCast<int16_t>(), (static_cast<std::int16_t>(value & 0xFF) << 8) | (value & 0xFF), (filled) >> 1);
                if(filled&1) buf.SetValue(filled-1,value);
            }
            else if constexpr (std::is_same_v<T, int16_t>||std::is_same_v<T, int32_t>) Duplicate<T>(buf, value, filled);
            else if constexpr (std::is_same_v<T, int64_t>){
                uint8_t tmp = (filled >> 5) + 1;
                Duplicate<int32_t>(buf.template ReinterpretCast<int32_t>(), static_cast<int32_t>((value >> 32) & 0xffffffff), maskhigh, tmp, 1, 8);
                Duplicate<int32_t>(buf.template ReinterpretCast<int32_t>(), static_cast<int32_t>(value & 0xffffffff), masklow, tmp, 1, 8);
            }
        }
        else //大于等于32KB的就不拷贝了 每次直接搬出一行 带宽影响不大
        {
            int bound = min(repeat * inner_elems,ub_buf_elems);
            // if(cur_elems%this->align_elems != 0){
            //     MyFillPad(vecbuf,filled);
            //     //pad之后 vecbuf一定32B对齐 但是只对较小数据采取措施
            //     //相对较大行原封不动
            // }
            while ((filled * 2) <= bound )//倍增不超界
            {
                MyCopy(buf, filled);
                filled *= 2;//倍增拷贝
            }
        }
        Queue.EnQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM,T>(buf);
        return filled;
    }

    // 修正后的 copyOut：多行/单行模式区分
    __aicore__ inline void copyOut(int32_t out_base_elem,
        int32_t offset_elems,     // 行内偏移
        int32_t inner_elems,     // 一行的元素数 (in)
        int32_t fill_elems,      // UB 中有效元素数
        int32_t repeat,          // 需要的总行数
        int32_t cur_chunk_elems) // 本 chunk 的列数 (cur)
    {
        int32_t rows_in_vec = max(1,fill_elems / inner_elems); // 可以保证fill是inner的整倍数
        auto vecbuf = Queue.DeQue<T>();
        int32_t copy_width = min (cur_chunk_elems,inner_elems);
        int32_t rows_done = 0;
        while (rows_done < repeat)
        {
            int32_t this_rows = min(rows_in_vec, repeat - rows_done);
            int32_t dst_pos = out_base_elem + rows_done * inner_elems + offset_elems;
            MyDataCopyPadOut(vecbuf,dst_pos, this_rows*copy_width);
            rows_done += this_rows;
        }
        Queue.FreeTensor(vecbuf);
    }
    __aicore__ inline void copyInMulti(int32_t in_base , int32_t inner_elems,int32_t step){
        auto buf = Queue.AllocTensor<T>();
        AscendC::DataCopy(buf, srcGm[in_base],inner_elems*step+pad);
        // 入队供 computeExpandVec 取出
        Queue.EnQue(buf);
    }
    __aicore__ inline void copyOutMulti(int32_t out_base , int32_t inner_elems,int32_t step,int32_t repeat){
        auto buf = Queue.DeQue<T>();
        DataCopyExtParams cp{static_cast<uint16_t>(step),static_cast<uint32_t>(inner_elems*sizeof(T)),0,static_cast<uint32_t>(inner_elems*(repeat-1)*sizeof(T)),0};
        for(int offset=0;offset<repeat;offset++)
            AscendC::DataCopyPad(dstGm[out_base + offset*inner_elems], buf,cp);
        // 入队供 computeExpandVec 取出
        Queue.FreeTensor(buf);
    }
    

    __aicore__ inline void performTileBroadcast(int tile_idx,int32_t outer_index,int32_t step = 1){
        int32_t inner_elems = inner[tile_idx]; // in
        int32_t repeat = repeater[tile_idx];   // repeat
        int32_t in_base = compute_in_base(tile_idx, outer_index);
        int32_t out_base = compute_out_base(tile_idx, outer_index);

        int32_t chunk_elems = min(ub_buf_elems, inner_elems);
        int32_t num_chunks = (inner_elems + chunk_elems - 1) / chunk_elems;
        int threads_idx = GetBlockIdx()/outer[tile_idx];//0，6，12，18，24，30，36的 idx分别为0,1,2,3,4,5,6
        int threads_num = (GetBlockNum()-outer_index+this->outer[tile_idx]-1)/this->outer[tile_idx];
        if(step>1){
            
            DataCopyExtParams cp_in{1,static_cast<uint32_t>(step*inner_elems*sizeof(T)),0,0,0};
            DataCopyExtParams cp_out{static_cast<uint16_t>(step),static_cast<uint32_t>(inner_elems*sizeof(T)),0,static_cast<uint32_t>(0+(repeat-1)*inner_elems*sizeof(T)),0};
            auto buf = Queue.AllocTensor<T>();
            DataCopyPad(buf,srcGm[in_base],cp_in,{0,0,0,0});
            Queue.EnQue<T>(buf);
            Queue.DeQue<T>();
            for(int i=0;i<repeat;i++)
                DataCopyPad(dstGm[out_base+i*inner_elems],buf,cp_out);
            Queue.FreeTensor<T>(buf);
        }else
        if(num_chunks>=threads_num){
            for (int32_t c = threads_idx; c < num_chunks; c+=threads_num)
            {
                int32_t offset = c * chunk_elems;
                int32_t cur = min(chunk_elems, inner_elems - offset);

                // 1) 读一个行片段或整行
                copyIn(in_base, offset, cur);
                // 2) 计算填充（可能倍增多行；仅当 cur == inner 且 full row）
                int32_t fill_elems = computeExpandVec(cur, inner_elems, repeat);
                // // 3) 写出
                copyOut(out_base, offset, inner_elems, fill_elems, repeat, cur);
            }
        }else{
            //线程需要继续下沉
            int32_t c = threads_idx%num_chunks;//任务索引
            int32_t offset = c * chunk_elems;
            int32_t cur = min(chunk_elems, inner_elems - offset);

            int threads_idx_2 = threads_idx / num_chunks;//任务下线程偏移
            int threads_num_2 = (threads_num - c + num_chunks-1) / num_chunks;//该任务分到线程数
            // 1) 读一个行片段或整行
            copyIn(in_base, offset, cur);
            // 2) 计算填充（可能倍增多行；仅当 cur == inner 且 full row）
            int32_t fill_elems = computeExpandVec(cur, inner_elems, repeat);
            //线程下沉到任务， 因此 具有相同输入仅输出不同。

            copyOut(out_base, offset, inner_elems, fill_elems, repeat, cur);
        }
    }
    __aicore__ inline void MyDataCopyPadOut(LocalTensor<T> & buf,int32_t dst_offset, int32_t elems)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(elems * sizeof(T)), 0, 0, 0};
        DataCopyPad(dstGm[dst_offset], buf,copyParams);
    }
    __aicore__ inline void MyDataCopyPadOutMultiRow(LocalTensor<T> & buf,int32_t dst_offset, int32_t rows, int32_t inner_elems,uint32_t step=1)
    {
        DataCopyExtParams copyParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(inner_elems * sizeof(T)), 1, step, 0};
        DataCopyPad(dstGm[dst_offset], buf,copyParams); 
    }

private:
    AscendC::GlobalTensor<T> srcGm;
    AscendC::GlobalTensor<T> dstGm;
    AscendC::GlobalTensor<T> wsGm;
    AscendC::GlobalTensor<int32_t> dst32Gm;
    AscendC::GlobalTensor<int32_t> ws32Gm;

    AscendC::TPipe *pipe;
    
    AscendC::TQueBind<AscendC::TPosition::VECIN,AscendC::TPosition::VECOUT,BUFFER_NUM> Queue;
    GM_ADDR ws_addr;
    GM_ADDR src_addr;
    GM_ADDR dst_addr;
    
    int32_t blockIdx;
    int32_t blockStride;
    int32_t num_cores;

    int32_t num_tile_entries;
    int32_t outer[4];
    int32_t repeater[4];
    int32_t inner[4];
    int32_t outputsize;
    int32_t tiling_size;
    int32_t total_outer_repeats;
    int32_t dtype_bytes;
    int32_t ub_buf_elems;
    int32_t align_elems;
    const int pad =(32/sizeof(T)) - 1; 
    uint64_t masklow[1] = {0x5555555555555555};  // 0101...
    uint64_t maskhigh[1] = {0xaaaaaaaaaaaaaaaa}; // 1010...
    uint64_t mask16[2] = {0xffffffffffffffff, 0xffffffffffffffff};
    uint64_t mask32[1] = {0xffffffffffffffff};
    uint64_t mask64[1] = {0xffffffff};
};



extern "C" __global__ __aicore__ void hypot(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
     KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0); // 增加这一行
    
        
    if(TILING_KEY_IS(3)){
        assert(0);
        KernelExpand<int_of_bytes_t<sizeof(DTYPE_X)>> op1;
        op1.Init(x,z,workspace,tilingData,&pipe,false);
        op1.Process();
        pipe.Reset();
        op1.Init(y,workspace,workspace+5120,tilingData,&pipe,true);
        op1.Process();
        pipe.Reset();
        KernelHypot<DTYPE_X> op;
        op.Init(z,workspace,z, tilingData, &pipe);
        op.Process();
    }else if(TILING_KEY_IS(1)){
        // printf("need broadcast\n");
        KernelExpand<int_of_bytes_t<sizeof(DTYPE_X)>> op1;
        op1.Init(x,z,workspace,tilingData,&pipe,false);
        op1.Process();
        pipe.Reset();
        
        // KernelHypot<DTYPE_X> op;
        // op.Init(z, y, z, tilingData, &pipe);
        // op.Process();
    }else if(TILING_KEY_IS(2)){
        assert(0);
        // printf("need broadcast\n");
        KernelExpand<int_of_bytes_t<sizeof(DTYPE_X)>> op1;
        op1.Init(y,z,workspace,tilingData,&pipe,true);
        op1.Process();
        
        pipe.Reset();
        
        KernelHypot<DTYPE_X> op;
        op.Init(x, z, z, tilingData, &pipe);
        op.Process();
    
    }
    else if(TILING_KEY_IS(0)){
        // printf("no need broadcast\n");
        KernelHypot<DTYPE_X> op;
        op.Init(x, y, z, tilingData, &pipe);
        op.Process();
    }
    
    
}
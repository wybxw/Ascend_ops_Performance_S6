// ...existing code...
#include "hypot_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
const int blockDim = 40;


namespace optiling {

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        
        TilingData tiling;
        const gert::StorageShape* x1_shape = context->GetInputShape(0);
        const gert::StorageShape* x2_shape = context->GetInputShape(1);
        int32_t shapeX1[16];
        int32_t shapeX2[16];
        for(int i=0;i<15;i++)
            shapeX1[i]=shapeX2[i]=1;
        int64_t x1_sz=1;
        int64_t x2_sz=1;
        int flag1=0,flag2=0;
        int32_t dim = std::max(x1_shape->GetStorageShape().GetDimNum(),x2_shape->GetStorageShape().GetDimNum());
        int32_t dim_x1=x1_shape->GetStorageShape().GetDimNum();
        int32_t dim_x2=x2_shape->GetStorageShape().GetDimNum();
        for(int i=0;i<dim_x1;i++)
            shapeX1[dim_x1-1-i] = x1_shape->GetStorageShape().GetDim(i);//右对齐
        for(int i=0;i<dim_x2;i++)
            shapeX2[dim_x2-1-i] = x2_shape->GetStorageShape().GetDim(i);
        for (int i = 0; i < dim; i++){
            x1_sz *= shapeX1[i];
            x2_sz *= shapeX2[i];
            if(shapeX1[i]>shapeX2[i])
                flag1=1;
            if(shapeX2[i]>shapeX1[i])
                flag2=1;
        }
        if(flag1&&flag2){
            context->SetTilingKey(3);
        }
        else if(x1_sz>x2_sz){
            context->SetTilingKey(2);
        }
        else if(x1_sz<x2_sz){
            context->SetTilingKey(1);
            // return ge::GRAPH_FAILED;
        }
        else context->SetTilingKey(0);
        context->SetBlockDim(40);
        // 获取输入数据类型（改为支持 float/half/bfloat）
        ge::DataType inputDataType = context->GetInputDesc(0)->GetDataType();
        int dataTypeSize = 0;
        switch (inputDataType) {
            case ge::DT_FLOAT16:    dataTypeSize = 2; break;
            case ge::DT_BF16: dataTypeSize = 2; ;break;
            case ge::DT_FLOAT: dataTypeSize = 4; ;break;
            default:
                // 若非法类型，仍用 float 作为默认
                dataTypeSize = 4; break;
        }
        

        
        // printf("dim:%d\n",dim);
        // for(int i=0;i<dim;i++)
        //     printf("shapeX1[%d]:%d,shapeX2[%d]:%d\n",i,shapeX1[i],i,shapeX2[i]);
        int32_t outer_x1[4]={0};
        int32_t inner_x1[4]={0};
        int32_t repeater_x1[4]={0};
        int64_t output_size_x1 = 1;
        int64_t data_sz_x1 = 1;
        for (int i = 0; i < dim; i++)
            data_sz_x1 *= shapeX1[i],
            output_size_x1 *= std::max(shapeX1[i],shapeX2[i]);    
        // printf("data_sz:%lld,output_size:%lld\n",data_sz,output_size);
        int out_x1=data_sz_x1;
        int in_x1=1;
        int repeat_x1=1;
        int i_x1=0;
        int j_x1=0;
        while(i_x1<dim){
            if(shapeX1[i_x1]==1){
                repeat_x1*=std::max(shapeX1[i_x1],shapeX2[i_x1]);
            }else{
                if(repeat_x1!=1){
                    outer_x1[j_x1]=out_x1;
                    inner_x1[j_x1]=in_x1;
                    repeater_x1[j_x1]=repeat_x1;
                    in_x1*=repeater_x1[j_x1];
                    j_x1++;
                }
                repeat_x1=1;
                out_x1/=shapeX1[i_x1];
                in_x1*=shapeX1[i_x1];
            }
            ++i_x1;
        }
        if(repeat_x1!=1){
          outer_x1[j_x1]=out_x1;
          inner_x1[j_x1]=in_x1;
          repeater_x1[j_x1]=repeat_x1;
          j_x1++;
        }
        // for(int k=0;k<j_x1;k++)
        //     printf("outer:%d,repeat:%d.inner:%d\n",outer_x1[k],repeater_x1[k],inner_x1[k]);
        tiling.set_Expandsize_x1(j_x1);
        tiling.set_outer_x1(outer_x1);
        tiling.set_inner_x1(inner_x1);
        tiling.set_repeater_x1(repeater_x1);




        // printf("dim:%d\n",dim);
        // for(int i=0;i<dim;i++)
        //     printf("shapeX1[%d]:%d,shapeX2[%d]:%d\n",i,shapeX1[i],i,shapeX2[i]);
        int32_t outer_x2[4]={0};
        int32_t inner_x2[4]={0};
        int32_t repeater_x2[4]={0};
        int64_t output_size_x2 = 1;
        int64_t data_sz_x2 = 1;
        for (int i = 0; i < dim; i++)
            data_sz_x2 *= shapeX2[i],
            output_size_x2 *= std::max(shapeX1[i],shapeX2[i]);  
        // printf("data_sz:%lld,output_size:%lld\n",data_sz_x2,output_size_x2);
        int out_x2=data_sz_x2;
        int in_x2=1;
        int repeat_x2=1;
        int i_x2=0;
        int j_x2=0;
        while(i_x2<dim){
            if(shapeX2[i_x2]==1){
                repeat_x2*=std::max(shapeX1[i_x2],shapeX2[i_x2]);
            }else{
                if(repeat_x2!=1){
                    outer_x2[j_x2]=out_x2;
                    inner_x2[j_x2]=in_x2;
                    repeater_x2[j_x2]=repeat_x2;
                    in_x2*=repeater_x2[j_x2];
                    j_x2++;
                    repeat_x2=1;
                }
                out_x2/=shapeX2[i_x2];
                in_x2*=shapeX2[i_x2];
            }
            ++i_x2;
        }
        if(repeat_x2!=1){
          outer_x2[j_x2]=out_x2;
          inner_x2[j_x2]=in_x2;
          repeater_x2[j_x2]=repeat_x2;
          j_x2++;
        }
        tiling.set_totalLength(output_size_x2);
        tiling.set_Expandsize_x2(j_x2);
        tiling.set_outer_x2(outer_x2);
        tiling.set_inner_x2(inner_x2);
        tiling.set_repeater_x2(repeater_x2);
        
        size_t usrSize = 5120;
        if(j_x1>1) usrSize += output_size_x1 * dataTypeSize;
        if(j_x2>1) usrSize += output_size_x2 * dataTypeSize;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize() + 5120;
        size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
        currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
        
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
        return ge::GRAPH_SUCCESS;
    }
    
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x2_shape = context->GetInputShape(1);
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(std::max(x2_shape->GetDimNum(),x1_shape->GetDimNum()));
    int dim_x1=x1_shape->GetDimNum();
    int dim_x2=x2_shape->GetDimNum();
    int dim=std::max(dim_x1,dim_x2);
    for(int i=0;i<dim;i++){
        int x1=x1_shape->GetDim(dim_x1-1-i);
        if(dim_x1-1-i<0) x1=1;
        int x2=x2_shape->GetDim(dim_x2-1-i);
        if(dim_x2-1-i<0) x2=1;
        y_shape->SetDim(dim-1-i,std::max(x1,x2));
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Hypot : public OpDef {
public:
    explicit Hypot(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
            // .Follow("x");
        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Hypot);
}

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
      TILING_DATA_FIELD_DEF_ARR(int32_t, 4, outer_x1);
      TILING_DATA_FIELD_DEF_ARR(int32_t, 4, repeater_x1);
      TILING_DATA_FIELD_DEF_ARR(int32_t, 4, inner_x1);
      TILING_DATA_FIELD_DEF_ARR(int32_t, 4, outer_x2);
      TILING_DATA_FIELD_DEF_ARR(int32_t, 4, repeater_x2);
      TILING_DATA_FIELD_DEF_ARR(int32_t, 4, inner_x2);
      
      TILING_DATA_FIELD_DEF(int64_t, totalLength);
      TILING_DATA_FIELD_DEF(int32_t,Expandsize_x1);
      TILING_DATA_FIELD_DEF(int32_t,Expandsize_x2);
      TILING_DATA_FIELD_DEF(int8_t,dim); //维度
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Hypot,TilingData)
}

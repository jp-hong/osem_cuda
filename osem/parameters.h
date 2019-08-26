#ifndef _PARAMETERS
#define _PARAMETERS

#include <cmath>

#define INPUT_DIR "D:\\Data\\proj_cuda\\"
#define OUTPUT_DIR "D:\\Data\\OSEM test\\"

//#define INPUT_DIR "D:\\Data\\test\\"
//#define OUTPUT_DIR "D:\\Data\\SART test\\"

#define SAVE_FILE_NAME "OSEM_test_"
#define SAVE_INTERVAL 20

#define ITER 100
#define N_SUBSET 6
#define LAMDA 1.0f
#define REG_FAC 1.0f

#define PI 3.1415926535f

#define NX 128
#define NY NX
#define NZ 128

//#define NX 512
//#define NY NX
//#define NZ 512

#define NS 360
#define R_NS 360
#define SLEN PI * 2.f
#define S0 .0f

#define NU 256
#define DU 4.0f
#define ULEN NU * DU
#define U0 -ULEN / 2.f

#define NV 200
#define DV 4.0f
#define VLEN NV * DV
#define V0 -VLEN / 2.f

//#define NU 1024
//#define DU 0.398f
//#define ULEN NU * DU
//#define U0 -ULEN / 2.f
//
//#define NV 1024
//#define DV 0.398f
//#define VLEN NV * DV
//#define V0 -VLEN / 2.f

#define R 1100.0f
#define D 1500.0f

#define XLEN 460.f
#define YLEN 460.f
#define ZLEN 460.f

//#define R 1000.0f
//#define D 1500.0f
//
//#define XLEN 2.f * R * sinf(atanf(ULEN / 2.f / D))
//#define YLEN XLEN
//#define ZLEN 2.f * R * sinf(atanf(VLEN / 2.f / D))

#define DX XLEN / NX
#define DY YLEN / NY
#define DZ ZLEN / NZ

#define X0 -XLEN / 2.f
#define Y0 -YLEN / 2.f
#define Z0 -ZLEN / 2.f

#define IMAGE_LEN NX * NY * NZ
#define IMAGE_BYTES IMAGE_LEN * sizeof(float)

#define PROJ_LEN NU * NV
#define PROJ_BYTES PROJ_LEN * sizeof(float)

#endif
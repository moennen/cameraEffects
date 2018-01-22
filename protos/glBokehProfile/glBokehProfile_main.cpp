/*! *****************************************************************************
 *   \file glBokehProfile_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-01-13
 *   *****************************************************************************/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <glm/glm.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <omp.h>

#include <GL/glut.h>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace glm;

namespace
{
const string keys =
    "{help h usage ? |            | print this message   }"
    "{@abCurves      |          | abCurves filename    }"
    "{height         |512       | height of the bokeh profile texture}"
    "{width          |1024       | width of the bokeh profile texture}";

inline dvec3 wl_to_xyz( float wl )
{
   static const double cie_colour_match[81][3] = {
       {0.0014, 0.0000, 0.0065}, {0.0022, 0.0001, 0.0105}, {0.0042, 0.0001, 0.0201},
       {0.0076, 0.0002, 0.0362}, {0.0143, 0.0004, 0.0679}, {0.0232, 0.0006, 0.1102},
       {0.0435, 0.0012, 0.2074}, {0.0776, 0.0022, 0.3713}, {0.1344, 0.0040, 0.6456},
       {0.2148, 0.0073, 1.0391}, {0.2839, 0.0116, 1.3856}, {0.3285, 0.0168, 1.6230},
       {0.3483, 0.0230, 1.7471}, {0.3481, 0.0298, 1.7826}, {0.3362, 0.0380, 1.7721},
       {0.3187, 0.0480, 1.7441}, {0.2908, 0.0600, 1.6692}, {0.2511, 0.0739, 1.5281},
       {0.1954, 0.0910, 1.2876}, {0.1421, 0.1126, 1.0419}, {0.0956, 0.1390, 0.8130},
       {0.0580, 0.1693, 0.6162}, {0.0320, 0.2080, 0.4652}, {0.0147, 0.2586, 0.3533},
       {0.0049, 0.3230, 0.2720}, {0.0024, 0.4073, 0.2123}, {0.0093, 0.5030, 0.1582},
       {0.0291, 0.6082, 0.1117}, {0.0633, 0.7100, 0.0782}, {0.1096, 0.7932, 0.0573},
       {0.1655, 0.8620, 0.0422}, {0.2257, 0.9149, 0.0298}, {0.2904, 0.9540, 0.0203},
       {0.3597, 0.9803, 0.0134}, {0.4334, 0.9950, 0.0087}, {0.5121, 1.0000, 0.0057},
       {0.5945, 0.9950, 0.0039}, {0.6784, 0.9786, 0.0027}, {0.7621, 0.9520, 0.0021},
       {0.8425, 0.9154, 0.0018}, {0.9163, 0.8700, 0.0017}, {0.9786, 0.8163, 0.0014},
       {1.0263, 0.7570, 0.0011}, {1.0567, 0.6949, 0.0010}, {1.0622, 0.6310, 0.0008},
       {1.0456, 0.5668, 0.0006}, {1.0026, 0.5030, 0.0003}, {0.9384, 0.4412, 0.0002},
       {0.8544, 0.3810, 0.0002}, {0.7514, 0.3210, 0.0001}, {0.6424, 0.2650, 0.0000},
       {0.5419, 0.2170, 0.0000}, {0.4479, 0.1750, 0.0000}, {0.3608, 0.1382, 0.0000},
       {0.2835, 0.1070, 0.0000}, {0.2187, 0.0816, 0.0000}, {0.1649, 0.0610, 0.0000},
       {0.1212, 0.0446, 0.0000}, {0.0874, 0.0320, 0.0000}, {0.0636, 0.0232, 0.0000},
       {0.0468, 0.0170, 0.0000}, {0.0329, 0.0119, 0.0000}, {0.0227, 0.0082, 0.0000},
       {0.0158, 0.0057, 0.0000}, {0.0114, 0.0041, 0.0000}, {0.0081, 0.0029, 0.0000},
       {0.0058, 0.0021, 0.0000}, {0.0041, 0.0015, 0.0000}, {0.0029, 0.0010, 0.0000},
       {0.0020, 0.0007, 0.0000}, {0.0014, 0.0005, 0.0000}, {0.0010, 0.0004, 0.0000},
       {0.0007, 0.0002, 0.0000}, {0.0005, 0.0002, 0.0000}, {0.0003, 0.0001, 0.0000},
       {0.0002, 0.0001, 0.0000}, {0.0002, 0.0001, 0.0000}, {0.0001, 0.0000, 0.0000},
       {0.0001, 0.0000, 0.0000}, {0.0001, 0.0000, 0.0000}, {0.0000, 0.0000, 0.0000}};

   static const double startWl = 380.0f;
   static const double endWl = 780.0f;
   const double lerp = ( ( wl - startWl ) / ( endWl - startWl ) );
   const double currIdx = mix( 0.0f, 80.0f, lerp );

   const size_t pCurrI = std::floor( currIdx );
   const vec3 pXYZ(
       cie_colour_match[pCurrI][0], cie_colour_match[pCurrI][1], cie_colour_match[pCurrI][2] );

   const size_t nCurrI = std::ceil( currIdx );
   const vec3 nXYZ(
       cie_colour_match[nCurrI][0], cie_colour_match[nCurrI][1], cie_colour_match[nCurrI][2] );

   dvec3 xyz = mix( pXYZ, nXYZ, currIdx - std::floor( currIdx ) );

   return xyz;
}

struct colourSystem
{
   char* name;         /* Colour system name */
   double xRed, yRed,  /* Red x, y */
       xGreen, yGreen, /* Green x, y */
       xBlue, yBlue,   /* Blue x, y */
       xWhite, yWhite, /* White point x, y */
       gamma;          /* Gamma correction for system */
};
#define IlluminantC 0.3101, 0.3162         /* For NTSC television */
#define IlluminantD65 0.3127, 0.3291       /* For EBU and SMPTE */
#define IlluminantE 0.33333333, 0.33333333 /* CIE equal-energy illuminant */
#define GAMMA_REC709 0 /* Rec. 709 */

static struct colourSystem
    /* Name                  xRed    yRed    xGreen  yGreen  xBlue  yBlue    White point Gamma   */
    NTSCsystem = {"NTSC", 0.67, 0.33, 0.21, 0.71, 0.14, 0.08, IlluminantC, GAMMA_REC709},
    EBUsystem =
        {"EBU (PAL/SECAM)", 0.64, 0.33, 0.29, 0.60, 0.15, 0.06, IlluminantD65, GAMMA_REC709},
    SMPTEsystem = {"SMPTE", 0.630, 0.340, 0.310, 0.595, 0.155, 0.070, IlluminantD65, GAMMA_REC709},
    HDTVsystem = {"HDTV", 0.670, 0.330, 0.210, 0.710, 0.150, 0.060, IlluminantD65, GAMMA_REC709},
    CIEsystem = {"CIE", 0.7355, 0.2645, 0.2658, 0.7243, 0.1669, 0.0085, IlluminantE, GAMMA_REC709},
    Rec709system = {"CIE REC 709", 0.64, 0.33, 0.30, 0.60, 0.15, 0.06, IlluminantD65, GAMMA_REC709};

inline int constrain_rgb(double *r, double *g, double *b)
{
    double w;

    /* Amount of white needed is w = - min(0, *r, *g, *b) */

    w = (0 < *r) ? 0 : *r;
    w = (w < *g) ? w : *g;
    w = (w < *b) ? w : *b;
    w = -w;

    /* Add just enough white to make r, g, b all positive. */

    if (w > 0) {
        *r += w;  *g += w; *b += w;
        return 1;                     /* Colour modified to fit RGB gamut */
    }

    return 0;                         /* Colour within RGB gamut */
}

void norm_rgb(double *r, double *g, double *b)
{
#define Max(a, b)   (((a) > (b)) ? (a) : (b))
    double greatest = Max(*r, Max(*g, *b));

    if (greatest > 0) {
        *r /= greatest;
        *g /= greatest;
        *b /= greatest;
    }
#undef Max
}

inline vec3 xyz_to_rgb( const dvec3 xyz )
{
   colourSystem* cs = &CIEsystem;

   double xc = xyz.x;
   double yc = xyz.y;
   double zc = xyz.z;

   double xr, yr, zr, xg, yg, zg, xb, yb, zb;
   double xw, yw, zw;
   double rx, ry, rz, gx, gy, gz, bx, by, bz;
   double rw, gw, bw;

   xr = cs->xRed;
   yr = cs->yRed;
   zr = 1 - ( xr + yr );
   xg = cs->xGreen;
   yg = cs->yGreen;
   zg = 1 - ( xg + yg );
   xb = cs->xBlue;
   yb = cs->yBlue;
   zb = 1 - ( xb + yb );

   xw = cs->xWhite;
   yw = cs->yWhite;
   zw = 1 - ( xw + yw );

   /* xyz -> rgb matrix, before scaling to white. */

   rx = ( yg * zb ) - ( yb * zg );
   ry = ( xb * zg ) - ( xg * zb );
   rz = ( xg * yb ) - ( xb * yg );
   gx = ( yb * zr ) - ( yr * zb );
   gy = ( xr * zb ) - ( xb * zr );
   gz = ( xb * yr ) - ( xr * yb );
   bx = ( yr * zg ) - ( yg * zr );
   by = ( xg * zr ) - ( xr * zg );
   bz = ( xr * yg ) - ( xg * yr );

   /* White scaling factors.
      Dividing by yw scales the white luminance to unity, as conventional. */

   rw = ( ( rx * xw ) + ( ry * yw ) + ( rz * zw ) ) / yw;
   gw = ( ( gx * xw ) + ( gy * yw ) + ( gz * zw ) ) / yw;
   bw = ( ( bx * xw ) + ( by * yw ) + ( bz * zw ) ) / yw;

   /* xyz -> rgb matrix, correctly scaled to white. */

   rx = rx / rw;
   ry = ry / rw;
   rz = rz / rw;
   gx = gx / gw;
   gy = gy / gw;
   gz = gz / gw;
   bx = bx / bw;
   by = by / bw;
   bz = bz / bw;

   /* rgb of the desired point */
   double r = ( rx * xc ) + ( ry * yc ) + ( rz * zc );
   double g = ( gx * xc ) + ( gy * yc ) + ( gz * zc );
   double b = ( bx * xc ) + ( by * yc ) + ( bz * zc );

   constrain_rgb(&r, &g, &b);
   norm_rgb(&r, &g, &b);

   vec3 rgb;
   rgb.r = r;
   rgb.g = g;
   rgb.b = b;

   return rgb;
}

double bbTemp = 5700;                  /* Hidden temperature argument
                                         to BB_SPECTRUM. */
double bb_spectrum(double wavelength)
{
    double wlm = wavelength * 1e-9;   /* Wavelength in meters */

    return (3.74183e-16 * pow(wlm, -5.0)) /
           (exp(1.4388e-2 / (wlm * bbTemp)) - 1.0);
}


/*inline vec3 xyz_to_rgb( const vec3 xyz )
{
   mat3 rgb2xyz = {{2.3706743f, -0.5138850f, 0.0052982f},
                   {-0.9000405f, 1.4253036f, -0.0146949f},
                   {-0.4706338f, 0.0885814f, 1.0093968f}};

   /*mat3 rgb2xyz = {2.3706743f,
                   -0.9000405f,
                   -0.4706338f,
                   -0.5138850f,
                   1.4253036f,
                   0.0885814f,
                   0.0052982f,
                   -0.0146949f,
                   1.0093968f};*/

 /*  return rgb2xyz * xyz;
}*/

// this represent a set of curves from lens height to focal length bias
struct AbModel
{
   float minWl;
   float maxWl;

   size_t nWlSamples;
   size_t nHeightSamples;

   size_t texHeight;
   size_t texWidth;

   size_t height;
   size_t focalLength;

   float maxSphericalAberration;
   float maxChromaticAberration;

   // xy = foc dist of the 2 intersecting envelop points
   // zw = scaling factor at 0,x
   vec4 remappingModel;

   AbModel(
       const float minW = 380.0f,
       const float maxW = 780.0f,
       const size_t nWS = 32,
       const size_t nHS = 128,
       const size_t h = 512,
       const size_t w = 1024 )
       : minWl( minW ),
         maxWl( maxW ),
         nWlSamples( nWS ),
         nHeightSamples( nHS ),
         texHeight( h ),
         texWidth( w ),
         height( h / 2 ),
         focalLength( w / 2 ),
         maxSphericalAberration(0.5),
         maxChromaticAberration(0.25)
   {
        // compute the remapping model : a 3 piece-wise linear model of the stretching
        const float maxNormHeight(static_cast<float>(height)/texHeight);
	const float focNorm(static_cast<float>(focalLength/texWidth));
        const vec2 frontLineEnv(-maxNormHeight/focNorm,maxNormHeight);
	const float minFocNorm((focalLength*(1.0f-maxSphericalAberration-maxChromaticAberration))/texWidth);
        const vec2 backLineEnv(maxNormHeight/minFocNorm,-maxNormHeight);
        const float backFrontEnvIntersectX = (backLineEnv.y-frontLineEnv.y)/(frontLineEnv.x-backLineEnv.x);
        const vec2 backFrontEnvIntersect(backFrontEnvIntersectX,frontLineEnv.x*backFrontEnvIntersectX+frontLineEnv.y); 
        const vec2 backBoundEnvIntersect((backLineEnv.x-1.0f)/(-backLineEnv.y),1.0f);
        
	remappingModel.x = backFrontEnvIntersect.x * texWidth;
	remappingModel.y = backBoundEnvIntersect.x * texWidth;
	remappingModel.z = 1.0f / maxNormHeight;
        remappingModel.w = 1.0f / backFrontEnvIntersect.y  ;  
   }

   float getFocDistance( const float h, const float wl )
   {
      const float normCAb = clamp( ( maxWl - wl ) / ( maxWl - minWl ), 0.0f, 1.0f );
      const float sqCAb = mix( 0.0f, maxChromaticAberration, normCAb /** normCAb*/ );
      const float normSAb = clamp( ( height - h ) / height, 0.0f, 1.0f );
      const float sqSAb = mix( maxSphericalAberration, 0.0f, normSAb /* * normSAb*/ );

      return focalLength * ( 1.0 - sqSAb - sqCAb );
   }

   inline float getStretchHeight(const float foc)
   { 
      	
      
   }

   inline vec2 toGl( const float f, const float h )
   {
      return vec2( 2.0f * ( f / texWidth ) - 1.0f, 2.0f * ( h / texHeight ) - 1.0f );
   }

} abModel;

float getFocDst( const float focSrc, const float heightDst, const vec2 dir )
{
   return focSrc + heightDst * dir.x / dir.y;
}

void drawBokehProfile( void )
{
   // sample the height and draw the line to the corresponding foci points
   /// NB : the profile is symetric thus we draw a line from the starting lens
   ///      height down to the foci point and from the foci point up to the
   ///      intersection with the max height line

   ///
   glClear( GL_COLOR_BUFFER_BIT );
   glEnable( GL_BLEND );
   glBlendFunc( GL_CONSTANT_COLOR, GL_ONE );
   // glBlendFunc( GL_ONE, GL_ONE );

   dvec3 XYZ( 0.0 );
   double normSpec = 0.0;
   for ( size_t swl = 0; swl < abModel.nWlSamples; ++swl )
   {
      const float wl = mix( abModel.minWl, abModel.maxWl, (float)swl / ( abModel.nWlSamples - 1 ) );
      const dvec3 xyz = bb_spectrum(wl) * wl_to_xyz( wl );
      XYZ += xyz;
      normSpec += bb_spectrum(wl) ;

   }
   normSpec = abModel.nWlSamples / normSpec;
   const double normXYZ = 1.0 / ( XYZ.x + XYZ.y + XYZ.z );

   vec3 RGB = xyz_to_rgb( XYZ );

   cout << "White Point : " << RGB.x << " " << RGB.y << " " << RGB.z  << endl;


   for ( size_t swl = 0; swl < abModel.nWlSamples; ++swl )
   {
      const float wl = mix( abModel.minWl, abModel.maxWl, (float)swl / ( abModel.nWlSamples - 1 ) );
      const dvec3 xyz = (bb_spectrum(wl) * normXYZ) * wl_to_xyz( wl ) ;
      const vec3 rgb = xyz_to_rgb( xyz );

      cout << wl << " " << bb_spectrum(wl) * normSpec << endl;

      const float sampleZ = (bb_spectrum(wl) * normSpec) / abModel.nHeightSamples;
      glBlendColor( sampleZ, sampleZ, sampleZ, 1.0f );
   

      //cout << wl << " -> " << rgb.x << " " << rgb.y << " " << rgb.z << endl;
 
      glColor3fv( value_ptr( rgb ) );

      for ( size_t sh = 0; sh < abModel.nHeightSamples; ++sh )
      {
         glBegin( GL_LINE_STRIP );
         
         const float h =
             mix( 0.0f, (float)abModel.height, (float)sh / ( abModel.nHeightSamples - 1 ) );
         
         {
            const vec2 pd = abModel.toGl( 0.0, h );//*abModel.remappingModel.z;         
            glVertex2f( pd.x, pd.y );
         }
         
         const float foc = abModel.getFocDistance( h, wl );
	
         if (foc > abModel.remappingModel.x )
         {
            // point to the intersection
            vec2 p(abModel.remappingModel.x, h-h*abModel.remappingModel.x/foc); 
	    vec2 pd = abModel.toGl(p.x,p.y);//*abModel.remappingModel.w;
            glVertex2f( pd.x, pd.y );
	    if ( foc > abModel.remappingModel.y  )
            {
               p = vec2(abModel.remappingModel.y, h-h*abModel.remappingModel.y/foc); 
	       pd = abModel.toGl(p.x,p.y);//1.0f;
	       glVertex2f( pd.x, pd.y );
	       p = vec2( foc, 0.0 ); 
	       pd = abModel.toGl(p.x,p.y);//1.0f;
	       glVertex2f( pd.x, pd.y );
            }
            else
            {
	       p = vec2( foc, 0.0 ); 
	       pd = abModel.toGl(p.x,p.y);//*mix(abModel.remappingModel.w,1.0,(foc-abModel.remappingModel.x)/(abModel.remappingModel.y-abModel.remappingModel.x));
	       glVertex2f( pd.x, pd.y );
            }            
         }	
	 else
         {
            vec2 pd = abModel.toGl( foc, 0.0 );//*mix(abModel.remappingModel.z,abModel.remappingModel.w,foc/abModel.remappingModel.x);
            glVertex2f( pd.x, pd.y );
            p = abModel.toGl(abModel.remappingModel.x, -h+h*abModel.remappingModel.x/foc);//*abModel.remappingModel.w;
            glVertex2f( pd.x, pd.y );
         } 
                 const float b = getFocDst( foc, abModel.texHeight, normalize( vec2( foc, h ) ) );
         const vec2 e = abModel.toGl( b, abModel.texHeight );

         glVertex2f( m.x, m.y );
         glVertex2f( e.x, e.y );
         glEnd();
      }
   }

   glFlush();
}
}

int main( int argc, char** argv )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }
   const string abCurvesFilename = parser.get<string>( "@abCurves" );
   const int width = parser.get<int>( "width" );
   const int height = parser.get<int>( "height" );

   abModel = AbModel( 380.0f, 780.0f, 128, height/10.0f, height, width );

   glutInit( &argc, argv );
   glutInitDisplayMode( GLUT_SINGLE );
   glutInitWindowSize( width, height );
   glutInitWindowPosition( 100, 100 );
   glutCreateWindow( "Bokeh Profile" );
   glutDisplayFunc( drawBokehProfile );
   glutMainLoop();

   return ( 0 );
}

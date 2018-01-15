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

// this represent a set of curves from lens height to focal length bias
struct AbCurves
{
   vector< vector<vec2> > data;

   bool generateSimpleSpherical(const size_t heightRes, const size_t focRes)
   {
	data.resize(1);
	data[0].resize(heightRes);
	for(auto&& pt : data[])
	{
		pt.x =
                pt.y =
	}
	
   }

} curves;

void drawBokehProfile(void)
{
    // sample the height and draw the line to the corresponding foci points
    /// NB : the profile is symetric thus we draw a line from the starting lens
    ///      height down to the foci point and from the foci point up to the 
    ///      intersection with the max height line 
   
    /// 
     
    glClear(GL_COLOR_BUFFER_BIT);
    for (auto && wl : curves.wavelength)
    { 
       glColor();
       for (float h=0.0; h<maxHeight; h+=heightStep)
       { 
          glBegin(GL_LINES);
          glVertex3f(0.0, 0.0, 0.0);
          glVertex3f(0.5, 0.0, 0.0);
          glVertex3f(0.5, 0.5, 0.0);
          glVertex3f(0.0, 0.5, 0.0);
          glEnd();
       }
    }
    glFlush();
}

}

int main(int argc, char** argv)
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }
   const string abCurvesFilename = parser.get<string>( "@abCurves" );
   const int width = parser.get<int>("width"); 
   const int height = parser.get<int>("height"); 

   curves.generateSimpleSpherical(height,width);	
   
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_SINGLE);
   glutInitWindowSize(width, height);
   glutInitWindowPosition(100, 100);
   glutCreateWindow("Bokeh Profile");
   glutDisplayFunc(drawBokehProfile);
   glutMainLoop();
   
   return ( 0 );
}



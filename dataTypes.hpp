#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <vector>

class ImageData {

   public:

      // **** Constructors ****
      ImageData() {}

      ImageData(cv::Mat i) : image(i) { 

         // Detect keypoints
         cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
         f2d->detect(image, keypoints);

         // Calculate descriptors
         f2d->compute(image, keypoints, descriptors);
      }


      // **** Functions **** 
      cv::Mat&                   getImage()               { return image;               }
      std::vector<cv::KeyPoint>& getKeyPoints()           { return keypoints;           }
      cv::Mat&                   getDescriptors()         { return descriptors;         }
      cv::Point                  getKeyPointAt(int index) { return keypoints[index].pt; }
      cv::Point                  getTL()                  { return tl;                  }
      void                       setTL(cv::Point t)       { tl = t;                     }
      

   private:

      cv::Mat image;
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;
      cv::Point tl;

};

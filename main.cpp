#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include "dataTypes.hpp"

int detect_car(ImageData& previous, ImageData& current, std::string filename);
void generatePossibleBoxes(ImageData& previous, std::vector<cv::Point>& boxes);
void printDistribution(std::vector<cv::DMatch> matches, ImageData& previous, ImageData& current);

// Constants
const int max_image        = 253;   // Total number of images in the sequence
const int min_displacement = -1;    // (min number of pixels)^2 car moved
const int max_displacement = 40;    // (max number of pixels)^2 car moved
const int rect_width       = 100;   // fixed size of car detecting square
const int image_width      = 640;   // width of image
const int image_height     = 272;   // height of image
const float max_dist_bonus = 10;    // maximum bias towards centre pixels
const float dist_numerator = 50;    // numerator for distance weighting

int main(int argc, const char* argv[]) {

   // Load images
   ImageData previous;

   for (int image_index = 1; image_index < max_image; ++image_index) {
      // Generate file name
      std::stringstream file_index;
      file_index << std::setfill('0') << std::setw(8) << image_index;
      std::string filename ("../data/");
      filename.append(file_index.str());
      filename.append(".jpg");
      std::cout << filename << std::endl;

      // Load image
      const cv::Mat image = cv::imread(filename, 1);
      if (image.empty()) {
         std::cout << "can't load image " << filename << std::endl;
         return 0;
      }

      // Create and store image information
      ImageData current(image);

      // Find the car
      if (image_index > 1) {
         detect_car(previous, current, filename);
      }

      else current.setTL(cv::Point(0,120));

      previous = current;
   }

   return 0;
}

// Attempt to detect the car and draw it to the screen
// Filename required only for saving the image
int detect_car(ImageData& previous, ImageData& current, std::string filename) {

   // Match decripters using FLANN
   cv::FlannBasedMatcher matcher;
   std::vector<cv::DMatch> matches;
   matcher.match(previous.getDescriptors(), current.getDescriptors(), matches);
   //printDistribution(matches, previous, current);

   // Cull matches to keep only good ones
   std::vector<cv::DMatch> good_matches;
   cv::Point prev_tl = previous.getTL();
   cv::Point prev_br (prev_tl.x+rect_width, prev_tl.y+rect_width);
   for (auto m = matches.begin(); m != matches.end(); ++m) {

      cv::Point p1 = previous.getKeyPointAt(m->queryIdx);
      cv::Point p2 = current.getKeyPointAt(m->trainIdx);

      bool same_box = false;
      if (p1.x > prev_tl.x && p1.x < prev_br.x && p1.y > prev_tl.y && p1.y < prev_br.y) {
         same_box = true;
      }

      float d2 = (p2.y-p1.y)*(p2.y-p1.y) + (p2.x-p1.x)*(p2.x-p1.x);
      if (d2 > min_displacement && d2 < max_displacement && same_box) {
         good_matches.push_back(*m);
      }
   }

   // Try to shuffle the previous tracking box to keep up with car
   cv::Point best_tl(0,0);
   cv::Point best_br(0,0);
   std::vector<cv::Point> possible_boxes;
   generatePossibleBoxes(previous, possible_boxes);
   int max_pixel_count = -1;
   for (auto box = possible_boxes.begin(); box != possible_boxes.end(); ++box) {
      
      cv::Point tl = *box;
      cv::Point br (std::min(tl.x + rect_width, image_width),
                    std::min(tl.y + rect_width, image_height));
      cv::Point centre (tl.x + rect_width/2, tl.y + rect_width/2);
      float pixel_count = 0;
      float distance_bonus = 0;

      // Calculate how many points are inside this rectangle
      for (auto n = good_matches.begin(); n != good_matches.end(); ++n) {
         cv::Point p = current.getKeyPointAt(n->trainIdx);
         if (p.x > tl.x && p.x < br.x && p.y > tl.y && p.y < br.y) {
            ++pixel_count;

            // Prefer points near the centre
            int distance = sqrt((centre.y-p.y)*(centre.y-p.y) + (centre.x-p.x)*(centre.x-p.x));
            distance_bonus += std::min((float)dist_numerator / (float)distance, max_dist_bonus);
         }
      }
      pixel_count += distance_bonus;

      if (pixel_count > max_pixel_count) {
         max_pixel_count = pixel_count;
         best_tl = tl;
         best_br = br;
      }
   }

   // Save car detection result for next iteration
   current.setTL(best_tl);


   // Draw matches
   cv::Mat image_matches;
   cv::drawMatches(previous.getImage(), previous.getKeyPoints(),
                   current.getImage(), current.getKeyPoints(),
                   good_matches, image_matches, cv::Scalar::all(-1), 2);
   //image_matches = current.getImage();   // Used to show just image, no debug
   cv::rectangle(image_matches, best_tl, best_br, cv::Scalar(200,0,0), 4);


   // Add results to image and save.
   //cv::Mat output;
   //cv::drawKeypoints(input, keypoints_1, output);
   filename = filename.substr(3); // remove "../"
   filename = filename.substr(0,14); // remove .jpg
   filename.append ("png"); // remove .jpg
   //cv::imwrite(filename, image_matches);   // Used to save the picture files

   // Display window and wait for keyboard press
   cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
   cv::imshow( "Display window", image_matches); 
   cv::waitKey(0);

   return 0;
}

// Generate potential new car locations, very simple look up, down, left, right
void generatePossibleBoxes(ImageData& previous, std::vector<cv::Point>& boxes) {
   
   cv::Point tl = previous.getTL();
   boxes.push_back(tl);
   for (int i = -4; i <= 4; i += 2) {
      int x = std::min(std::max(0, tl.x + i), image_width);

      for (int j = -4; j <= 4; j += 2) {
         int y = std::min(std::max(0, tl.y + j), image_height);

         boxes.push_back(cv::Point(x, y));
      }
   }
}

// Debugging function to print out the distribution of feature movement (in pixel space)
void printDistribution(std::vector<cv::DMatch> matches, ImageData& previous, ImageData& current) {

   const int dist_size = 200;
   std::vector<int> distribution (dist_size, 0);

   for (auto m = matches.begin(); m != matches.end(); ++m) {

      cv::Point p1 = previous.getKeyPointAt(m->queryIdx);
      cv::Point p2 = current.getKeyPointAt(m->trainIdx);
      float d2 = (p2.y-p1.y)*(p2.y-p1.y) + (p2.x-p1.x)*(p2.x-p1.x);

      // Tally distribution
      int index = d2;
      if (index > (dist_size-1)) index = dist_size-1;
      ++distribution[index];
   }

   for (int i = 0; i < dist_size; ++i) {
      std::cout << i << " = " << distribution[i] << std::endl;
   }
}

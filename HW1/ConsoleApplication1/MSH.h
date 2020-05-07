#pragma once

#include "ImageProcess.h"
#include <iostream>
#include <string>
#include <vector>

class MSH {
public:
	class FeaturePoint {
	public:
		float* descriptor;
		float x;
		float y;
		int scale; // pyrimid height
		float cornerResponse;
		FeaturePoint(float _x, float _y, int _scale);
		float distance_sqr(FeaturePoint& fp);
	};

public:
	std::string dir; // 圖片資料夾路徑
	std::string subtitle; // 圖片資料夾路徑
	float focalLength; // 圖片焦距

	std::vector<cv::Mat> images; // 讀取進來的圖片
	std::vector<std::vector<FeaturePoint>> features;

	cv::Point2f Project(cv::Point2f p);
	cv::Point2f Unproject(cv::Point2f p);

	void Detect(int pyrimidHeight, int N, std::vector<cv::Mat>& pyrimid, std::vector<FeaturePoint>& features);
	void HarrisConerDetector(cv::Mat img, int scale, std::vector<FeaturePoint>& features);
	void Describe(std::vector<cv::Mat>& pyrimid, std::vector<FeaturePoint>& features);
	void Match(int i0, int i1, std::vector<std::pair<FeaturePoint*, FeaturePoint*>>& feature_pairs);
	cv::Point2f Align(std::vector<std::pair<FeaturePoint*, FeaturePoint*>>& feature_pairs);
	cv::Mat Pano(std::vector<cv::Point2f>& offsets);


public: 

	MSH(std::string dir, std::string subtitle, float focalLength);
	~MSH();

	cv::Mat GetImage(int idx);
	cv::Mat GetImageFeature(int idx);
	cv::Mat GetImageMatch(int i0, int i1);

	cv::Mat Stitching(int pyrimidHeight, int N);

	void ShowInputs();


};
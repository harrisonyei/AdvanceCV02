#include "MSH.h"
#include "fileIO.h"

#include <algorithm>

using namespace cv;
using namespace std;

#define WINDOW_SIZE 41

#define GRAY(_R_, _G_, _B_) (((_R_) * 0.27) + ((_G_) * 0.67) + ((_B_) * 0.06))

#define LOOP_MAT(__mat__) for(int row=0;row<(__mat__).rows;row++)\
                                for(int col=0;col<(__mat__).cols;col++)

MSH::MSH(string dir, string subtitle, float focalLength)
{
    this->dir = dir;
    this->subtitle = subtitle;
	this->focalLength = focalLength;

    vector<string> filenames = get_filenames_in(dir, subtitle);

    this->images.clear();

    for (string& filename : filenames) {
        Mat image;
        
        image = imread(filename, IMREAD_COLOR);
        if (!image.data)// Check for invalid input
        {
            cout << "Could not open or find the image : " << filename << std::endl;
            continue;
        }

        this->images.push_back(image);
    }

}

MSH::~MSH()
{
}

cv::Mat MSH::GetImage(int idx)
{
    if (idx >= images.size()) {
        cout << " Index Out of Range\n";
        return cv::Mat();
    }

    return images[idx];
}

cv::Mat MSH::GetImageFeature(int idx)
{
	if (idx >= images.size()) {
		cout << " Index Out of Range\n";
		return cv::Mat();
	}

	cv::Mat result = images[idx].clone();

	for (auto& fp : this->features[idx]) {
		float scale = pow(2, fp.scale);
		Point p(fp.x / scale, fp.y / scale);
		cv::circle(result, p, 3, Scalar(0, 0, 255));
	}

	return result;
}

cv::Mat MSH::GetImageMatch(int i0, int i1)
{
	cv::Mat result;
	vector<Mat> mats = { images[i0] ,images[i1] };
	hconcat(mats, result);

	int offsetX = images[i0].cols;

	for (auto& fp : this->features[i0]) {
		float scale = pow(2, fp.scale);
		Point p(fp.x / scale, fp.y / scale);
		cv::circle(result, p, 3, Scalar(0,0,255));
	}

	for (auto& fp : this->features[i1]) {
		float scale = pow(2, fp.scale);
		Point p(fp.x / scale + offsetX, fp.y / scale);
		cv::circle(result, p, 3, Scalar(0, 0, 255));
	}


	for (auto& f0 : this->features[i0]) {
		float m0 = INT_MAX;
		float m1 = INT_MAX;

		FeaturePoint* mf = nullptr;

		for (auto& f1 : this->features[i1]) {
			float dist_sqr = f0.distance_sqr(f1);

			if (m0 > dist_sqr) {
				if (m1 > m0) {
					m1 = m0;
				}
				m0 = dist_sqr;
				mf = &f1;
			}
			else if (m1 > dist_sqr) {
				m1 = dist_sqr;
			}
		}

		if ((m1 == 0 || (m0 / m1) < 0.7) && mf != nullptr) {
			float s0 = pow(2, f0.scale);
			float s1 = pow(2, mf->scale);

			Point p0(f0.x / s0, f0.y / s0);
			Point p1(mf->x / s1 + offsetX, mf->y / s1);
			cv::line(result, p0, p1, Scalar(0, 255, 100));
		}

	}
	return result;
}

cv::Mat MSH::Stitching(int pyrimidHeight, int N)
{
	this->features.clear();

	for (int idx = 0; idx < this->images.size(); idx++) {
		cout << "Detect Features... " << idx+1 << "/" << this->images.size() << "       \r";
		Mat img = this->images[idx];

		Mat gray(img.rows, img.cols, CV_32F);
		LOOP_MAT(img) {
			cv::Vec3b& color = img.at<cv::Vec3b>(row, col);
			gray.at<float>(row, col) = GRAY(color[2], color[1], color[0]);
		}

		// build gaussian pyrimid & feature detect
		vector<Mat> pyrimid;

		Mat pyrImg = gray.clone();
		pyrimid.push_back(pyrImg.clone());

		for (int i = 1; i < pyrimidHeight; i++) {
			pyrUp(pyrImg, pyrImg, Size(pyrImg.cols * 2, pyrImg.rows * 2));
			pyrimid.push_back(pyrImg.clone());
		}

		this->features.push_back(vector<FeaturePoint>());

		Detect(pyrimidHeight, N, pyrimid, this->features.back());

		Describe(pyrimid, this->features.back());
	}
	cout << "\nDone." << endl;

	vector<Point2f> offsets;
	for (int idx = 0; idx < this->images.size(); idx++) {
		cout << "Align Images... " << idx+1 << "/" << this->images.size() << "       \r";

		vector<pair<FeaturePoint*, FeaturePoint*>> feature_pairs;
		Match(idx, (idx + 1) % this->images.size(), feature_pairs);

		Point2f offset = Align(feature_pairs);
		offsets.push_back(offset);
	}
	cout << "\nDone." << endl;

	cout << "\n Generating Panorama..." << endl;
	Mat pano = Pano(offsets);

	return pano;
}

cv::Point2f MSH::Project(cv::Point2f p)
{
	float theta = atan2(p.x, this->focalLength);
	float h = p.y / (sqrt(p.x * p.x + this->focalLength * this->focalLength));
	return cv::Point2f(theta, h);
}

cv::Point2f MSH::Unproject(cv::Point2f p)
{
	//float theta = p.x;
	//float h = p.y;
	float x = tan(p.x) * this->focalLength;
	float y = p.y * sqrt(x * x + this->focalLength * this->focalLength);
	return cv::Point2f(x, y);
}

// sort ascending
bool compareListFeature( MSH::FeaturePoint& f0, MSH::FeaturePoint& f1)
{
	return (f0.cornerResponse < f1.cornerResponse);
}
void MSH::Detect(int pyrimidHeight, int N, std::vector<cv::Mat>& pyrimid, std::vector<FeaturePoint>& features)
{
	vector<FeaturePoint> tmpFeatures;
	for (int i = 0; i < pyrimidHeight; i++) {
		HarrisConerDetector(pyrimid[i], i, tmpFeatures);
	}

	// sort ascending
	sort(tmpFeatures.begin(), tmpFeatures.end(), compareListFeature);

	// Non maximal suppression
	if (tmpFeatures.size() > N) {

		vector<Point2f> grid[10][10];
		float w = 10.0f / pyrimid[0].cols;
		float h = 10.0f / pyrimid[0].rows;

		auto GridInsert = [&grid, &w, &h](Point2f& p) {
			int x = p.x * w;
			int y = p.y * h;

			if (x < 0) x = 0; else if (x > 9) x = 9;
			if (y < 0) y = 0; else if (y > 9) y = 9;

			grid[y][x].push_back(p);
		};

		auto GridFind = [&grid, &w, &h](Point2f& p, float r) {
			int minX = (p.x - r) * w;
			int maxX = (p.x + r) * w;
			int minY = (p.y - r) * h;
			int maxY = (p.y + r) * h;

			if (minX < 0) minX = 0; else if (minX > 9) minX = 9;
			if (maxX < 0) maxX = 0; else if (maxX > 9) maxX = 9;
			if (minY < 0) minY = 0; else if (minY > 9) minY = 9;
			if (maxY < 0) maxY = 0; else if (maxY > 9) maxY = 9;

			float minDist = INT_MAX;
			for (int x = minX; x <= maxX; x++) {
				for (int y = minY; y <= maxY; y++) {
					for(Point2f& fp : grid[y][x]){
						float distance_sqr = (fp.x - p.x) * (fp.x - p.x) + (fp.y - p.y) * (fp.y - p.y);
						if (distance_sqr < minDist) {
							minDist = distance_sqr;
						}
					}
				}
			}

			return minDist;
		};

		float r = (WINDOW_SIZE / 2.0f) + 4;
		float decay = 0.9f;

		while (N > 0)
		{
			bool found = false;
			for (int i = tmpFeatures.size() - 1; i >= 0; i--) {
				float f_s = pow(2, tmpFeatures[i].scale);
				Point2f f_p(tmpFeatures[i].x / f_s, tmpFeatures[i].y / f_s);

				float dist_sqr = GridFind(f_p, r);

				if (dist_sqr > (r * r)) {
					features.push_back(tmpFeatures[i]);
					GridInsert(f_p);
					tmpFeatures.erase(tmpFeatures.begin() + i);
					found = true;
					N -= 1;
					break;
				}
			}

			if (!found) {
				r *= decay;
			}
		}
	} else {
		features = tmpFeatures;
	}
}

void MSH::HarrisConerDetector(cv::Mat img, int scale, std::vector<FeaturePoint>& features)
{
	GaussianBlur(img, img, cv::Size(3, 3), 1, 1, BORDER_ISOLATED);

	cv::Mat Ix(img.rows, img.cols, CV_32F);
	cv::Mat Iy(img.rows, img.cols, CV_32F);

	// Gradient X
	Sobel(img, Ix, CV_32F, 1, 0, 3);

	// Gradient Y
	Sobel(img, Iy, CV_32F, 0, 1, 3);

	cv::Mat Sxx = Ix.mul(Ix);
	cv::Mat Sxy = Ix.mul(Iy);
	cv::Mat Syy = Iy.mul(Iy);

	GaussianBlur(Sxx, Sxx, cv::Size(WINDOW_SIZE, WINDOW_SIZE), 2, 2, BORDER_ISOLATED);
	GaussianBlur(Sxy, Sxy, cv::Size(WINDOW_SIZE, WINDOW_SIZE), 2, 2, BORDER_ISOLATED);
	GaussianBlur(Syy, Syy, cv::Size(WINDOW_SIZE, WINDOW_SIZE), 2, 2, BORDER_ISOLATED);

	int count = 0;
	for (int row = 0; row < img.rows; row += 3)
		for(int col = 0; col < img.cols; col += 3)
		{
			FeaturePoint mFeature(row, col, scale);
			float maximumResponse = 0;

			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					int irow = i + row;
					int jcol = j + col;

					if (irow < 0 || irow >= Sxx.rows || jcol < 0 || jcol >= Sxx.cols) {
						continue;
					}

					float sxx = Sxx.at<float>(irow, jcol);
					float sxy = Sxy.at<float>(irow, jcol);
					float syy = Syy.at<float>(irow, jcol);

					float detM = sxx * syy - sxy * sxy;
					float traceM = sxx + syy;

					float cornerResponse = 0;

					if (traceM != 0) {
						cornerResponse = detM / traceM;
					}

					if (cornerResponse > 10 && cornerResponse > maximumResponse) {
						maximumResponse = cornerResponse;

						mFeature.x = jcol;
						mFeature.y = irow;
						mFeature.cornerResponse = cornerResponse;
					}
				}
			}
			
			if (maximumResponse > 0) {
				features.push_back(mFeature);
			}
		}

}

void MSH::Describe(std::vector<cv::Mat>& pyrimid, std::vector<FeaturePoint>& features)
{
	vector<Mat> Ixs;
	vector<Mat> Iys;

	for (auto& img : pyrimid) {
		cv::Mat blur;

		GaussianBlur(img, blur, cv::Size(WINDOW_SIZE, WINDOW_SIZE), 4, 4, BORDER_ISOLATED);

		cv::Mat Ix(img.rows, img.cols, CV_32F);
		cv::Mat Iy(img.rows, img.cols, CV_32F);

		// Gradient X
		Sobel(blur, Ix, CV_32F, 1, 0, 3);

		// Gradient Y
		Sobel(blur, Iy, CV_32F, 0, 1, 3);

		Ixs.push_back(Ix);
		Iys.push_back(Iy);
	}
	
	int half_window = WINDOW_SIZE / 2;
	int sample_step = WINDOW_SIZE / 8;
	for (auto& f : features) {
		Mat& Ix = Ixs[f.scale];
		Mat& Iy = Iys[f.scale];
		Mat& img = pyrimid[f.scale];

		float dx = Ix.at<float>(f.y, f.x);
		float dy = Iy.at<float>(f.y, f.x);

		float length = sqrt((dx * dx) + (dy * dy));

		if (length > 0) {
			dx /= length;
			dy /= length;
		}

		Point2f up(dx , dy);
		Point2f right(dy , -dx);

		f.descriptor = new float[64];

		// getsubpixel
		cv::Mat patch;
		Point2f pt;
		for (int x = -half_window, x_count = 0; x_count < 8; x += sample_step, x_count += 1) {
			for (int y = -half_window, y_count = 0; y_count < 8; y += sample_step, y_count += 1) {

				int idx = y_count * 8 + x_count;
				f.descriptor[idx] = 0;

				for (int i = 0; i < sample_step; i++) {
					for (int j = 0; j < sample_step; j++) {
						pt = (x + i) * right + (y + j) * up + Point2f(f.x, f.y);
						cv::getRectSubPix(img, cv::Size(1, 1), pt, patch);
						f.descriptor[idx] += patch.at<float>(0, 0);
					}
				}

				f.descriptor[idx] /= (sample_step * sample_step);

			}
		}
	}
}

void MSH::Match(int i0, int i1, vector<pair<FeaturePoint*, FeaturePoint*>>& feature_pairs)
{
	feature_pairs.clear();
	for (auto& f0 : this->features[i0]) {
		float m0 = INT_MAX;
		float m1 = INT_MAX;
		FeaturePoint* mf = nullptr;
		for (auto& f1 : this->features[i1]) {
			float dist_sqr = f0.distance_sqr(f1);
			if (m0 > dist_sqr) {
				if (m1 > m0) {
					m1 = m0;
				}
				m0 = dist_sqr;
				mf = &f1;
			}
			else if (m1 > dist_sqr) {
				m1 = dist_sqr;
			}
		}
		if ((m1 == 0 || (m0 / m1) < 0.65) && mf != nullptr) {
			feature_pairs.push_back(make_pair(&f0, mf));
		}
	}
}

cv::Point2f MSH::Align(std::vector<std::pair<FeaturePoint*, FeaturePoint*>>& feature_pairs)
{
	int height = this->images[0].rows;
	int width = this->images[0].cols;

	int halfH = height / 2;
	int halfW = width / 2;

	// Project
	std::vector<std::pair<Point2f, Point2f>> feature_loc_pairs;
	for (int i = 0; i < feature_pairs.size(); i++) {

		float s0 = pow(2, feature_pairs[i].first->scale);
		float s1 = pow(2, feature_pairs[i].second->scale);

		Point2f p0 = Project(Point2f(feature_pairs[i].first->x / s0 - halfW, feature_pairs[i].first->y / s0 - halfH));
		Point2f p1 = Project(Point2f(feature_pairs[i].second->x / s1 - halfW, feature_pairs[i].second->y / s1 - halfH));

		feature_loc_pairs.push_back(make_pair(p0, p1));
	}

	//RANSAC
	// P = 0.99
	// p = 0.075 assume picture overlap 30% and 25% chance of correct match
	// n = 1
	const int k = 60;

	// non-repeating random sample
	std::vector<int> sample_idx;
	for (int i = 0; i < feature_loc_pairs.size(); i++)
		sample_idx.push_back(i);

	int minError = INT_MAX;
	float threshold = 800;
	Point2f best;

	for (int i = 0; i < k && sample_idx.size() > 0; i++) {
		int idx = rand() % sample_idx.size();
		int sample = sample_idx[idx];

		sample_idx.erase(sample_idx.begin() + idx);

		Point2f& sp0 = feature_loc_pairs[sample].first;
		Point2f& sp1 = feature_loc_pairs[sample].second;

		float sm1 = sp0.x - sp1.x;
		float sm2 = sp0.y - sp1.y;

		float error = 0;
		for (int j = 0; j < feature_loc_pairs.size(); j++) {
			if (j != sample) {
				Point2f& p0 = feature_loc_pairs[j].first;
				Point2f& p1 = feature_loc_pairs[j].second;

				float m1 = p0.x - p1.x;
				float m2 = p0.y - p1.y;

				float dist = ((m1 - sm1) * (m1 - sm1) + (m2 - sm2) * (m2 - sm2));

				if (dist < threshold) {
					error += dist;
				}

			}
		}

		if (error < minError) {
			minError = error;
			best.x = sm1;
			best.y = sm2;
		}
	}

	return best;
}

cv::Mat MSH::Pano(std::vector<cv::Point2f>& offsets)
{
	int height = this->images[0].rows;
	int width = this->images[0].cols;

	int halfH = height / 2;
	int halfW = width / 2;

	Point2f upperCenter = Project(Point2f(0, halfH));
	Point2f middleRight = Project(Point2f(halfW, 0));

	double w_t_ratio = (double)middleRight.x / halfW;
	double h_h_ratio = (double)upperCenter.y / halfH;

	// generate wrapped images
	vector<Mat> wrappeds;
	for (int idx = 0; idx < this->images.size(); idx++) {
		Mat img = this->images[idx];
		Mat wrapped = img.clone();

		cv::Mat patch;
		LOOP_MAT(wrapped) {
			Point2f pt = Unproject(Point2f((col - halfW) * w_t_ratio, (row - halfH) * h_h_ratio));
			pt.x += halfW;
			pt.y += halfH;

			if (pt.x > wrapped.cols || pt.x < 0 || pt.y > wrapped.rows || pt.y < 0) {
				wrapped.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
			}
			else {
				cv::getRectSubPix(img, cv::Size(1, 1), pt, patch);
				Vec3b& color = patch.at<Vec3b>(0, 0);
				wrapped.at<Vec3b>(row, col) = color;
			}
		}

		wrappeds.push_back(wrapped);
	}

	// merge images
	// calculate final image size
	vector<Point2f> wrappedOffsets;
	Point2f accOffset(0, 0);
	float minX, minY, maxX, maxY;
	minX = minY = 0;
	maxX = maxY = 0;
	for (int idx = 0; idx < offsets.size()-1; idx++) {
		Point2f offset = Unproject(offsets[idx]);
		wrappedOffsets.push_back(offset);

		accOffset += offset;

		minX = min(minX, accOffset.x);
		minY = min(minY, accOffset.y);

		maxX = max(maxX, accOffset.x);
		maxY = max(maxY, accOffset.y);

	}

	cout << minX << " " << maxX << " " << minY << " " << maxY << endl;

	Mat pano(height + (maxY - minY), width + (maxX - minX), CV_8UC3);
	Point2f offset(0, 0);
	for (int idx = 0; idx < wrappeds.size(); idx++) {
		Mat& wrapped = wrappeds[idx];
		LOOP_MAT(wrapped) {
			int pano_col = col + offset.x - minX;
			int pano_row = row + offset.y - minY;

			if (pano_col >= 0 && pano_col < pano.cols && pano_row >= 0 && pano_row < pano.rows) {
				pano.at<Vec3b>(pano_row, pano_col) = wrapped.at<Vec3b>(row, col);
			}
		}
		offset += wrappedOffsets[idx];
	}

	return pano;
}

void MSH::ShowInputs()
{
    namedWindow("Display window", WINDOW_AUTOSIZE);
    for (int i = 0; i < this->images.size(); i++) {
        cv::Mat image = this->images[i];
        imshow("Display window " + to_string(i), image);                   // Show our image inside it.
    }
    waitKey(0);
}

MSH::FeaturePoint::FeaturePoint(float _x, float _y, int _scale)
{
	x = _x;
	y = _y;
	scale = _scale;
}

float MSH::FeaturePoint::distance_sqr(FeaturePoint& fp)
{
	if (descriptor == nullptr || fp.descriptor == nullptr) {
		cout << "descriptor is nullptr!!" << endl;
		throw -1;
	}

	float dist = 0;
	for (int i = 0; i < 64; i++) {
		dist += (descriptor[i] - fp.descriptor[i]) * (descriptor[i] - fp.descriptor[i]);
	}

	return dist;
}

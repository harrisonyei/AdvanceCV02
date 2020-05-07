

#include <iostream>
#include <string>

#include "MSH.h"

using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
    if (argc < 5) {
        cout << "Incorrect arguments! : [ Directory ] [ File Type(jpg, nef, png) ] [FOCAL LENGTH] [Output File]" << std::endl;
        return -1;
    }

	srand(time(NULL));

	string dir      = argv[1];
	string subtitle = argv[2];
	float focal = atof(argv[3]);

    string exportPath = argv[4];

	MSH* msh = new MSH(dir, subtitle, focal);

	Mat image = msh->Stitching(2, 250);
	imwrite(exportPath, image);

	image = msh->GetImageFeature(0);
	imwrite("Tmp/Feature_0.jpg", image);
	for (int i = 1; i < msh->images.size(); i++) {
		image = msh->GetImageFeature(i);
		imwrite("Tmp/Feature_" + to_string(i) + ".jpg", image);
		image = msh->GetImageMatch(i - 1, i);
		imwrite("Tmp/Match_" + to_string(i) + ".jpg", image);
	}

    //imshow("Display window", image);               // Show our image inside it.
    //waitKey(0);
    
    return 0;
}


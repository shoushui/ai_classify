#include "ClassifyOnnx.h"


std::string getnames(std::string imgNames) {
	std::string::size_type iPos = imgNames.find_last_of('\\') + 1;
	std::string filename = imgNames.substr(iPos, imgNames.length() - iPos);
	return filename;
}


int main() {
	std::string onnxPath = "./models/tl_hron_cls_efficientnet_416.onnx";
	std::string DirPath = "./images/tl_hron";
	std::string OutPath = "./results/tl_hron/";
	cv::Mat img;
	std::vector<cv::String> fileNames;
	cv::glob(DirPath, fileNames);

	Classify model;
	//double t = (double)cv::getTickCount();
	model.LoadModel(onnxPath);
	std::string label;
	for (int i = 0; i < fileNames.size(); i++) {
		std::cout << fileNames[i] << std::endl;
		img = cv::imread(fileNames[i]);
		std::string outName = getnames(fileNames[i]);
		double t0 = (double)cv::getTickCount();
		label = model.detect(img);

		cv::imwrite(OutPath + "/"+ label + "/" + outName, img);
	
		
		t0 = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
		std::cout << "Predict time in seconds: " << t0 << std::endl;
	}
	//t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	//std::cout << "all time in seconds: " << t << std::endl;
	return 0;
}
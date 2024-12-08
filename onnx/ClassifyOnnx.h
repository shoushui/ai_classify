#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>


class Classify {
public:
	Classify();
	void LoadModel(std::string modelpath);
	std::string detect(cv::Mat srcimg);

private:
	cv::Mat ResizeImage(cv::Mat srcimg, int* newh, int* neww, int* top, int* left);
	void normalize_(cv::Mat img, std::vector<float>& _inputImage_);

	int m_inpWidth;
	int m_inpHeight;
	int m_classnum;
	std::vector<std::string> m_classes = { "ng", "ok"};			//模型所有类别

	std::vector<char*> m_inputNames;  // 定义一个字符指针vector
	std::vector<char*> m_outputNames; // 定义一个字符指针vector

	std::vector<std::vector<int64_t> > m_inputNodeDims; // >=1 outputs  ，二维vector
	std::vector<std::vector<int64_t> > m_outputNodeDims; // >=1 outputs ,int64_t C/C++标准
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Classify"); // 初始化环境
	Ort::Session* ortSession = nullptr;    // 初始化Session指针选项
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();  //初始化Session对象
};

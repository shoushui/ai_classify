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
	std::vector<std::string> m_classes = { "ng", "ok"};			//ģ���������

	std::vector<char*> m_inputNames;  // ����һ���ַ�ָ��vector
	std::vector<char*> m_outputNames; // ����һ���ַ�ָ��vector

	std::vector<std::vector<int64_t> > m_inputNodeDims; // >=1 outputs  ����άvector
	std::vector<std::vector<int64_t> > m_outputNodeDims; // >=1 outputs ,int64_t C/C++��׼
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Classify"); // ��ʼ������
	Ort::Session* ortSession = nullptr;    // ��ʼ��Sessionָ��ѡ��
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();  //��ʼ��Session����
};

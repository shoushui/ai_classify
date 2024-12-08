#include "ClassifyOnnx.h"

Classify::Classify()
{
}

void Classify::LoadModel(std::string modelpath)
{
	std::vector<int> result_load;
	std::wstring widestr = std::wstring(modelpath.begin(), modelpath.end());  //用于UTF-16编码的字符

	//gpu, https://blog.csdn.net/weixin_44684139/article/details/123504222
	//CUDA加速开启
	//OrtCUDAProviderOptions cuda_options{0,OrtCudnnConvAlgoSearch::EXHAUSTIVE,std::numeric_limits<size_t>::max(),0,true};
	//sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
	//sessionOptions.SetInterOpNumThreads(10);
	//OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);



	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);  //设置图优化类型
	ortSession = new Ort::Session(env, widestr.c_str(), sessionOptions);  // 创建会话，把模型加载到内存中
	size_t numInputNodes = ortSession->GetInputCount();  //输入输出节点数量                         
	size_t numOutputNodes = ortSession->GetOutputCount();
	Ort::AllocatorWithDefaultOptions allocator;   // 配置输入输出节点内存
	for (int i = 0; i < numInputNodes; i++)
	{
		m_inputNames.push_back(ortSession->GetInputName(i, allocator));		// 内存
		Ort::TypeInfo input_type_info = ortSession->GetInputTypeInfo(i);   // 类型
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();  // 
		auto input_dims = input_tensor_info.GetShape();    // 输入shape
		m_inputNodeDims.push_back(input_dims);	// 保存
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		m_outputNames.push_back(ortSession->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ortSession->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		m_outputNodeDims.push_back(output_dims);
	}
	m_inpHeight = m_inputNodeDims[0][2];
	m_inpWidth = m_inputNodeDims[0][3];
	m_classnum = m_outputNodeDims[0][1];
}

cv::Mat Classify::ResizeImage(cv::Mat srcimg, int* newh, int* neww, int* top, int* left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = m_inpHeight;
	*neww = m_inpWidth;
	cv::Mat dstimg;
	if (srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->m_inpHeight;
			*neww = int(m_inpWidth / hw_scale);
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((m_inpWidth - *neww) * 0.5);
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, m_inpWidth - *neww - *left, cv::BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)m_inpHeight * hw_scale;
			*neww = m_inpWidth;
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*top = (int)(m_inpHeight - *newh) * 0.5;
			cv::copyMakeBorder(dstimg, dstimg, *top, m_inpHeight - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 114);
		}
	}
	else {
		cv::Size size = cv::Size(*neww, *newh);
		cv::resize(srcimg, dstimg, size, cv::INTER_AREA);
	}
	return dstimg;
}

void Classify::normalize_(cv::Mat img, std::vector<float>& _inputImage_)
{
	std::vector<float> mean_value{ 0.485, 0.456, 0.406 };
	std::vector<float> std_value{ 0.229, 0.224, 0.225 };
	int row = img.rows;
	int col = img.cols;
	_inputImage_.resize(row * col * img.channels());  // vector大小
	for (int c = 0; c < 3; c++)  // bgr
	{
		for (int i = 0; i < row; i++)  // 行
		{
			for (int j = 0; j < col; j++)  // 列
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];  // Mat里的ptr函数访问任意一行像素的首地址,2-c:表示rgb
				_inputImage_[c * row * col + i * col + j] = (pix / 255.0 - mean_value[c]) / std_value[c];

			}
		}
	}
}


std::string Classify::detect(cv::Mat srcimg)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;

	/*cv::Mat dstimg = ResizeImage(srcimg, &newh, &neww, &padh, &padw);*/
	cv::Mat dstimg;
	cv::resize(srcimg, dstimg, cv::Size(m_inpHeight, m_inpWidth));
	std::vector<float> _inputImage_;		// 输入图片
	normalize_(dstimg, _inputImage_);
	// 定义一个输入矩阵，int64_t是下面作为输入参数时的类型
	std::array<int64_t, 4> input_shape_{ 1, 3, this->m_inpHeight, this->m_inpWidth };

	//创建输入tensor
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, _inputImage_.data(), _inputImage_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	std::vector<Ort::Value> ort_outputs = ortSession->Run(Ort::RunOptions{ nullptr }, &m_inputNames[0], &input_tensor_, 1, m_outputNames.data(), m_outputNames.size());   // 开始推理
	float* floatarr = ort_outputs[0].GetTensorMutableData<float>(); // GetTensorMutableData

	// 得到最可能分类输出
	cv::Mat newarr = cv::Mat_<double>(1, m_classes.size()); //定义一个1*1000的矩阵
	for (int i = 0; i < newarr.rows; i++)
	{
		for (int j = 0; j < newarr.cols; j++) //矩阵列数循环
		{
			newarr.at<double>(i, j) = floatarr[j];
		}
	}
	for (int n = 0; n < newarr.rows; n++) {
		cv::Point classNumber;
		double classProb;
		cv::Mat probMat = newarr(cv::Rect(0, n, m_classes.size(), 1)).clone();
		cv::Mat result = probMat.reshape(1, 1);
		cv::minMaxLoc(result, NULL, &classProb, NULL, &classNumber);
		int classidx = classNumber.x;
		std::string label = m_classes.at(classidx);
		//printf("\n current image classification : %s, possible : %.2f\n", m_classes.at(classidx).c_str(), classProb);
		//cv::putText(srcimg, m_classes.at(classidx), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, 1);
		return label;
	}
}


#include <cv.h>
#include <highgui.h>
#include <cxcore.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <leveldb/db.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"

#include <string>
#include <fstream>

using namespace std;
using namespace cv;
using namespace caffe;
using std::vector;

#define PAIR_SIZE 16

int CreateDir(const char *sPathName, int beg) {
	char DirName[256];
	strcpy(DirName, sPathName);
	int i, len = strlen(DirName);
	if (DirName[len - 1] != '/')
		strcat(DirName, "/");

	len = strlen(DirName);

	for (i = beg; i < len; i++) {
		if (DirName[i] == '/') {
			DirName[i] = 0;
			if (access(DirName, 0) != 0) {
				CHECK(mkdir(DirName, 0755) == 0)<< "Failed to create folder "<< sPathName;
			}
			DirName[i] = '/';
		}
	}

	return 0;
}

char buf[101000];
int main(int argc, char** argv)
{

	//Caffe::set_phase(caffe::TEST);
	if (argc == 8 && strcmp(argv[7], "CPU") == 0) {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	} else {
		LOG(ERROR) << "Using GPU";
		Caffe::set_mode(Caffe::GPU);
	}

	Caffe::set_mode(Caffe::GPU);
        string gpunum(argv[5]);
	Caffe::SetDevice(gpunum[0] - '0');



	//NetParameter test_net_param;
	//ReadProtoFromTextFile(argv[1], &test_net_param);
	std::string test_net_param_s(argv[1]);
	Net<float> caffe_test_net(test_net_param_s, caffe::TEST);
	NetParameter trained_net_param;
	ReadProtoFromBinaryFile(argv[2], &trained_net_param);
	caffe_test_net.CopyTrainedLayersFrom(trained_net_param);


	string labelFile(argv[3]);
	int data_counts = 0, tcount = 0;;
	FILE * file = fopen(labelFile.c_str(), "r");
	while(fscanf(file, "%s%d", buf, &tcount ) > 0)
	{
		int id ;
		for(int i = 0; i < tcount; i ++)
		{
			fscanf(file, "%s%d", buf, &id );
		}
		data_counts++;
	}
	fclose(file);
	printf("sample number: %d\n", data_counts);

	vector<Blob<float>*> dummy_blob_input_vec;
	string rootfolder(argv[4]);
	rootfolder.append("/");
	CreateDir(rootfolder.c_str(), rootfolder.size() - 1);
	string folder;
	string fName;

	float output;
	int counts = 0;

	file = fopen(labelFile.c_str(), "r");

	Blob<float>* c1 = (*(caffe_test_net.top_vecs().rbegin()))[0];
    int c2 = c1->channels();
    printf("num:%d, channel:%d, height:%d, width:%d\n", c1->num(), c1->channels(), c1->height(), c1->width() );
	int batchCount = std::ceil( (float)(data_counts) / (floor)(c2) );//(test_net_param.layers(0).layer().batchsize()));//                (test_net_param.layers(0).layer().batchsize() ));

	string resulttxt = rootfolder + "videoResult.txt";
	FILE * resultfile = fopen(resulttxt.c_str(), "w");
	for (int batch_id = 0; batch_id < batchCount; ++batch_id)
	{
		LOG(INFO)<< "processing batch :" << batch_id+1 << "/" << batchCount <<"...";

		const vector<Blob<float>*>& result = caffe_test_net.Forward(dummy_blob_input_vec);
		Blob<float>* bboxs = (*(caffe_test_net.top_vecs().rbegin()))[0];
		int bsize = bboxs->num();
		int channels = bboxs->channels();
		int height = bboxs->height();
		int width = bboxs->width();
		printf("bsize: %d, channels: %d, height: %d, width: %d\n", bsize, channels, height, width);

		for (int i = 0; i < channels && counts < data_counts; i++, counts++)
		{
			char fname[1010];
			int id;
			fscanf(file, "%s%d", fname, &tcount ) ;
			for (int j = 0; j < tcount; j ++)
			{
				fscanf(file, "%s", fname);
				fscanf(file, "%d", &id);
			}
			fprintf(resultfile, "%s %d ", fname, id);

			for (int c = 0; c < bsize; c ++)
			{
				for(int h = 0; h < height; h ++)
					for(int w = 0; w < width; w ++)
						fprintf(resultfile, "%f ", (float)(bboxs->data_at(c, i, h, w)));
			}
			fprintf(resultfile, "\n");

		}

		// debug
		// break;
	}

	fclose(resultfile);
	fclose(file);

	return 0;
}

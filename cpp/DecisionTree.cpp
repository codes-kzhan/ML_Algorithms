#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <set>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#define random(x) ( rand() % x )
using namespace std;

struct TreeNode {
	int feature;
	float value;
	string maxClass;
	TreeNode* left = NULL;
	TreeNode* right = NULL;
	/*map<string, int> probas;*/

	TreeNode(int feature, float value) {
		this->feature = feature;
		this->value = value;
	}

	TreeNode(int feature, float value, string maxClass, TreeNode* left,
			TreeNode* right) {
		this->feature = feature;
		this->value = value;
		this->maxClass = maxClass;
		this->left = left;
		this->right = right;
	}
};

class BTree {

public:
	TreeNode* root;

	BTree( void ) {
		root = NULL;
	}

	bool isEmpty() {
		if (root == NULL)
			return true;
		else
			return false;
	}

	void preOrder(TreeNode* node) {
		if (node->left == NULL)
			cout << node->value << "\n";
		else {
			cout << node->value << "\n";
			preOrder(node->left);
			preOrder(node->right);
		}
	}

	int getNumLeafs( TreeNode* node ) {
		int numLeafs = 0;

		if (node->left == NULL)
			numLeafs += 1;

		else {
			numLeafs += getNumLeafs(node->left);
			numLeafs += getNumLeafs(node->right);
		}

		return numLeafs;
	}

	string getClass( vector<float> x, TreeNode* node ) {
		if ( node->left == NULL )
			return node->maxClass;
		else {
			if ( x[node->feature] > node->value )
				return getClass( x, node->right );
			else
				return getClass( x, node->left );
		}
	}
};

float correctNum( vector<string> yTrue, vector<string> yPred ) {
	float ret = 1.0;
	for ( unsigned int i = 0; i < yTrue.size(); i++ ) {
		const char *yt = yTrue[i].c_str();
		const char *yp = yPred[i].c_str();
		if ( strcmp( yt, yp ) == 0 )
			ret += 1.0;
	}
	return ret;
}

float gini(vector<string> y) {
	float ret = 0.0;
	map<string, float> yDict;

	for (unsigned int i = 0; i < y.size(); i++) {
		map<string, float>::iterator iter;
		iter = yDict.find(y[i]);
		if (iter == yDict.end()) {
			yDict[y[i]] = 0;
		}
		yDict[y[i]] += 1;
	}

	map<string, float>::iterator iter = yDict.begin();
	for (; iter != yDict.end(); ++iter) {
		float curr = iter->second;
		ret += (curr / y.size()) * (curr / y.size());
	}

	return 1 - ret;
}

string classify(vector<string> y) {
	map<string, float> yDict;

	for (unsigned int i = 0; i < y.size(); i++) {
		map<string, float>::iterator iter;
		iter = yDict.find(y[i]);
		if (iter == yDict.end()) {
			yDict[y[i]] = 0;
		}
		yDict[y[i]] += 1;
	}

	float max = 0.0;
	string maxId = "none";

	map<string, float>::iterator iter = yDict.begin();
	for (; iter != yDict.end(); ++iter) {
		float curr = iter->second;
		if (curr > max) {
			max = curr;
			maxId = iter->first;
		}
	}

	return maxId;
}

class DecisionTree {

private:

	float tol;
	unsigned int leastSample;
	int maxDepth;
	string merge;
	unsigned int maxFeatures;

public:

	BTree* tree;
	vector<string> targetList;

	DecisionTree(float tol, int leastSample, int maxDepth, string merge,
			int maxFeatures) {
		tree = new BTree();
		this->tol = tol;
		this->leastSample = leastSample;
		this->maxDepth = maxDepth;
		this->merge = merge;
		this->maxFeatures = maxFeatures;
	}

	DecisionTree( void ) {
		tree = new BTree();
		tol = 0.0001;
		leastSample = 1;
		maxDepth = 999;
		merge = "None";
		maxFeatures = 10;
	}

	void binSplitDataSet( vector<vector<float> > &mat0, vector<string> &y0,
			vector<vector<float> > &mat1, vector<string> &y1,
			vector<vector<float> > data, vector<string> y, int feature,
			float value ) {

		mat0.clear();
		y0.clear();
		mat1.clear();
		y1.clear();

		for (unsigned int i = 0; i < data.size(); i++) {
			vector<float> row = data[i];
			if (row[feature] <= value) {
				mat0.push_back(row);
				y0.push_back(y[i]);
			} else {
				mat1.push_back(row);
				y1.push_back(y[i]);
			}
		}
	}

	void chooseBestSplit( vector<vector<float> > data, vector<string> y,
			int &bestIndex, float &bestValue, string &maxClass ) {

		set<string> ySet( y.begin(), y.end() );
		vector<vector<float> > mat0;
		vector<string> y0;
		vector<vector<float> > mat1;
		vector<string> y1;

		if (ySet.size() == 1) {
			bestIndex = -1;
			bestValue = -1;
			maxClass = y[0];
			return;
		}

		int n = data[0].size();
		float S = gini(y);
		float bestS = 1.0;
		bestIndex = 0;
		bestValue = 0.0;
		vector<int> featureList;
		unsigned int visitedFeatures = 0;

		for (int i = 0; i < n; i++)
			featureList.push_back(i);

		while ( visitedFeatures < maxFeatures and featureList.size() > 0 ) {
			srand((int) time(0));
			int randNum = random( featureList.size() );
			int featIndex = featureList[randNum];
			vector<int>::iterator iter = featureList.begin();
			while (iter != featureList.end() && *iter != featIndex)
				iter++;
			if (iter != featureList.end())
				featureList.erase(iter);

			vector<float> column;
			for (unsigned int i = 0; i < data.size(); i++)
				column.push_back(data[i][featIndex]);

			set<float> featSet( column.begin(), column.end() );
			vector<float> featList;
			copy(featSet.begin(), featSet.end(), back_inserter(featList));
			if (featList.size() == 1)
				continue;

			visitedFeatures += 1;
			float lastS = 1.0;

			for (unsigned int i = 0; i < featList.size(); i++) {
				float splitVal = featList[i];
				binSplitDataSet(mat0, y0, mat1, y1, data, y, featIndex, splitVal);

				if (mat0.size() < leastSample || mat1.size() < leastSample)
					continue;

				float r0 = ( (float) y0.size() ) / y.size();
				float r1 = ( (float) y1.size() ) / y.size();
				float newS = r0 * gini(y0) + r1 * gini(y1);

				if ( newS > lastS )
					break;

				if ( newS < bestS ) {
					bestIndex = featIndex;
					bestValue = splitVal;
					bestS = newS;
				}

				lastS = newS;
			}
		}

		maxClass = classify(y);
		if (S - bestS < tol) {
			bestIndex = -1;
			bestValue = -1;
			return;
		}

		binSplitDataSet(mat0, y0, mat1, y1, data, y, bestIndex, bestValue);
		if (mat0.size() < leastSample || mat1.size() < leastSample) {
			bestIndex = -1;
			bestValue = -1;
			return;
		}
	}

	TreeNode* createTree( vector<vector<float> > data, vector<string> y, int depth ) {

		TreeNode* retTree = new TreeNode( -1, -1 );
		if (depth > maxDepth) {
			retTree->maxClass = classify(y);
			return retTree;
		}

		int bestIndex; float bestValue; string maxClass;
		chooseBestSplit( data, y, bestIndex, bestValue, maxClass );
		if ( bestIndex == -1 ) {
			retTree->maxClass = maxClass;
			return retTree;
		}

		vector<vector<float> > left;
		vector<string> yleft;
		vector<vector<float> > right;
		vector<string> yright;
		binSplitDataSet( left, yleft, right, yright, data, y, bestIndex, bestValue );
		retTree->feature = bestIndex;
		retTree->value = bestValue;
		retTree->maxClass = maxClass;
		retTree->left = createTree(left, yleft, depth + 1);
		retTree->right = createTree(right, yright, depth + 1);

		return retTree;
	}

	void train( vector<vector<float> > &data, vector<string> &y ) {

		set<string> targetSet( y.begin(), y.end() );
		copy(targetSet.begin(), targetSet.end(), back_inserter(targetList));
		long length = y.size();
		long cutPoint = length * 9 / 10;
		vector<string> ytrain;
		vector<vector<float> > train;
		vector<string> ytest;
		vector<vector<float> > test;

		/*for ( unsigned int i = 0; i < data.size(); i ++ ) {
			if (i < cutPoint) {
				train.push_back( data[i] );
				ytrain.push_back( y[i] );
			}

			else {
				test.push_back( data[i] );
				ytest.push_back(y[i]);
			}
		}*/

		if (merge == "None") {
			TreeNode* root = createTree(data, y, 0);
			tree->root = root;
		}
	}

	vector<string> predict( vector<vector<float> > data ) {
		vector<string> ret;
		for ( unsigned int i = 0; i < data.size(); i++ ) {
			vector<float> x = data[i];
			string pred = tree->getClass(x, tree->root);
			ret.push_back(pred);
		}
		return ret;
	}

	float score( vector<vector<float> > data, vector<string> y ) {
		vector<string> yPred = predict(data);
		return correctNum( y, yPred ) / (float) y.size();
	}
};

vector<string> splitEx(const string& src, string separate_character) {
	vector<string> strs;
	int separate_characterLen = separate_character.size();
	int lastPosition = 0, index = -1;
	while (-1 != (index = src.find(separate_character, lastPosition))) {
		strs.push_back(src.substr(lastPosition, index - lastPosition));
		lastPosition = index + separate_characterLen;
	}
	string lastString = src.substr(lastPosition); //截取最后一个分隔符后的内容
	if (!lastString.empty())
		strs.push_back(lastString); //如果最后一个分隔符后还有内容就入队
	return strs;
}

void readCSV(vector<vector<float> > &dataTrain, vector<string> &yTrain,
		vector<vector<float> > &dataTest, vector<string> &yTest,
		char* filename) {
	ifstream file(filename);
	string value;
	vector<string> strs;
	int i = 0;

	while (file.good()) {
		getline(file, value, '\n'); // read a string until next comma
		strs = splitEx(value, ",");
		int length = strs.size();

		if (i == 0) {
			i++;
			continue;
		} else if (i < 50000) {
			vector<float> row;
			string y;
			for (int j = 0; j < length; j++) {
				if (j < length - 1)
					row.push_back(atoi(strs[j].c_str()));
				else
					y = strs[j];
			}
			dataTrain.push_back(row);
			yTrain.push_back(y);
		} else {
			vector<float> row;
			string y;
			for (int j = 0; j < length; j++) {
				if (j < length - 1)
					row.push_back(atoi(strs[j].c_str()));
				else
					y = strs[j];
			}
			dataTest.push_back(row);
			yTest.push_back(y);
		}

		i++;
	}
	dataTest.pop_back();
	yTest.pop_back();
}

int main(int argc, char* argv[]) {

	char* filename = "/home/yejiming/desktop/python/ML_Algorithms/dataset.csv";
	vector<vector<float> > dataTrain;
	vector<string> yTrain;
	vector<vector<float> > dataTest;
	vector<string> yTest;
	readCSV(dataTrain, yTrain, dataTest, yTest, filename);

	DecisionTree learner;
	learner.train( dataTrain, yTrain );
	float s = learner.score( dataTest, yTest);
	cout << s;
}

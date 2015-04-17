#include <iostream>   
#include <string>   
#include<fstream> 
#include <vector>   
using namespace std;  

vector<string> splitEx(const string& src, string separate_character) {
	vector<string> strs;   
	int separate_characterLen = separate_character.size();
	int lastPosition = 0,index = -1; 
	while (-1 != (index = src.find(separate_character,lastPosition))) {   
        strs.push_back(src.substr(lastPosition,index - lastPosition));   
        lastPosition = index + separate_characterLen;   
    }   
    string lastString = src.substr(lastPosition);//截取最后一个分隔符后的内容   
    if (!lastString.empty())   
        strs.push_back(lastString);//如果最后一个分隔符后还有内容就入队   
    return strs; 
}

int main(int argc, char* argv[]) {  
  
  ifstream file ( "C:\\Users\\hp4230s\\Desktop\\行为分析\\CMHK第四期\\arrange2.csv" ); 
  string value; 
  vector<string> strs;
  int i = 0;
  
  while ( i<100 ) {  
     getline( file, value, '\n' ); // read a string until next comma
	 strs = splitEx(value, ",");
	 for (int i = 0; i < strs.size(); i++) {
		cout << strs[i] << ",";
	 }
	 i++;
  } 
}  

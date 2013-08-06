#pragma once

#include <iostream>

using namespace std;

class xmlDocument
{

private:
	string m_header, m_body, m_footer;
	
public:
	xmlDocument(void);
	~xmlDocument(void);
	string getHeader();
	string getBody();
	string getFooter();
	void addElement(string elementName, string elementValue);
};


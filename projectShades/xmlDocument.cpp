#include "xmlDocument.h"


xmlDocument::xmlDocument(void)
{
	m_header = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>\n\n<measurements>\n";
	m_body = "";
	m_footer = "</measurements>";
}


xmlDocument::~xmlDocument(void)
{
}

string xmlDocument::getHeader()
{
	return m_header;
}

string xmlDocument::getBody()
{
	return m_body;
}

string xmlDocument::getFooter()
{
	return m_footer;
}


void xmlDocument::addElement(string elementName, string elementValue)
{
	m_body += "\t<";
	m_body += elementName;
	m_body += ">\n";

	m_body += "\t\t";
	m_body += elementValue;

	m_body += "\n\t</";
	m_body += elementName;
	m_body += ">\n";
}
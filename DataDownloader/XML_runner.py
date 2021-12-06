import os
from DataDownloader.xml_utils import parseXML
for i, filename in enumerate(os.listdir("/Users/arnabgupta/Documents/NYU/Coursework/Fall 21/2590 - Natural Language Processing/Project/scientific-paper-summarisation/Data/XML_Papers/")):
    parseXML(os.path.join("/Users/arnabgupta/Documents/NYU/Coursework/Fall 21/2590 - Natural Language Processing/Project/scientific-paper-summarisation/Data/XML_Papers/" + filename), "/Users/arnabgupta/Documents/NYU/Coursework/Fall 21/2590 - Natural Language Processing/Project/scientific-paper-summarisation/Data/Parsed_Papers/")
    print("{}/{}".format(i+1, 10148))
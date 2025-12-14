from langchain_text_splitters import  RecursiveCharacterTextSplitter

text= '''
I would also like to thank my former Google colleagues, in particular the YouTube
video classification team, for teaching me so much about Machine Learning. I could
never have started the first edition without them. Special thanks to my personal ML
gurus: Clément Courbet, Julien Dubois, Mathias Kende, Daniel Kitachewsky, James
Pack, Alexander Pak, Anosh Raj, Vitor Sessak, Wiktor Tomczak, Ingrid von Glehn,
and Rich Washington. And thanks to everyone else I worked with at YouTube and in
the amazing Google research teams in Mountain View. Many thanks as well to Martin
Andrews, Sam Witteveen, and Jason Zaman for welcoming me into their Google
Developer Experts group in Singapore, with the kind support of Soonson Kwon, and
for all the great discussions we had about Deep Learning and TensorFlow. Anyone
interested in Deep Learning in Singapore should definitely join their Deep Learning
Singapore meetup. Jason deserves special thanks for sharing some of his TFLite
expertise for Chapter 19!
'''

#initialize the splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

chunks = splitter.split_text(text)
print(len(chunks))
print("chunks:", chunks)
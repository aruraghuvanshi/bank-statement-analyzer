# bank-statement-analyzer
Neural Network based bank statement analyzer

Currently operational for Axis, HDFC and Kotak Mahindra Bank formats (IND).
Code is pushed here for safe storage. 
Neural netowrk model weights are more than 255 MB and hence not uploaded.
Used TfIdfVectorizer and LabelEncoder to carry out supervised learning using ANN on the textual data descriptions on the bank statements and uses Tabula to extract information from the pdf files. A parallel code is developed for Pytesseract to extract the first page of the statement to extract Name, Bank and Period details using OpenCV to invert the image, detect the horizontal and vertical lines and create cell structure to extract information within the cells.
UI developed in REMI Python. 

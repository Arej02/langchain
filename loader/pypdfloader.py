# PypdfLoader along with splitter:
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter


loader=PyPDFLoader('Prac_Questions.pdf')

doc=loader.load() # Say our pdf had 5 pages, this will create 5 PDF object
# print(doc[1])

len_splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result=len_splitter.split_documents(doc) # All the 5 documents are split here
print(result[0].page_content)



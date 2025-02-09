import zipfile

def removeZipFiles(zipFileName):
    with zipfile.ZipFile(zipFileName, 'r') as zip_ref:
        zip_ref.extractall()
    return

removeZipFiles('P39-CS3-Data.zip')
removeZipFiles('P39-CS3-Python-Code.zip')
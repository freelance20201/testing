import os
import fitz  # pip install --upgrade pip; pip install --upgrade pymupdf
from tqdm import tqdm # pip install tqdm
from logging import exception
import pdf2image
import numpy as np
import PyPDF2
from PIL import Image
import layoutparser as lp
import io
from PIL import Image

def image_extractor(file_url):
  pdf_file= file_url


#reading the pdf file 
  obj_file = open(pdf_file,'rb')
  pdf_reader = PyPDF2.PdfFileReader(obj_file)


#init the model 
  model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                  extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                  label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

  #example = Pdf.open(pdf_file)

  #file = "/content/prem (1).pdf"
  pdf_file = fitz.open(file_url)

  for page_index in range(len(pdf_file)):
      # get the page itself
      page = pdf_file[page_index]
      image_list = page.get_images()
      
      # printing number of images found in this page
      if image_list:
          print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
      else:
          print("[!] No images found on page", page_index)
      for image_index, img in enumerate(page.get_images(), start=1):
          xref = img[0]
          width = img[3]
          height = img[2]
          base_image = pdf_file.extract_image(xref)
          image_bytes = base_image["image"]
          image_ext = base_image["ext"]
          image = Image.open(io.BytesIO(image_bytes))
          if((width>300 and height>500) and (page.rect.x0 > 0.0 and page.rect.y0 > 0.0)):
            image.save(open(f"image{page_index+1}_{image_index}.{image_ext}", "wb"))

  for i in range(pdf_reader.numPages):

          try:
              img = np.asarray(pdf2image.convert_from_path(file_url)[i])
              orig_img = img
              layout_result = model.detect(img)

              lp.draw_box(img, layout_result,  box_width=6, box_alpha=0.2, show_element_type=True)
              text_blocks = lp.Layout([b for b in layout_result if b.type=='Text'])
              lp.draw_box(img, text_blocks,  box_width=5, box_alpha=0.2, show_element_type=True, show_element_id=True)
              img_blocks = lp.Layout([b for b in layout_result if b.type=='Figure'])
              viz = lp.draw_box(img, img_blocks, box_width=5, box_alpha=0.2, show_element_type=True, show_element_id=True)

                #coverting into image 
              img = np.asarray(viz)

              img = Image.fromarray(orig_img)
              for j in range(len(img_blocks)):

                  img2 = img.crop(img_blocks[j].coordinates)
                  img2.save("output"+str([i][j])+".png")
            
        
          except:
              print('No image detected....')

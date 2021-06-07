#!/bin/bash
wget -c https://drive.google.com/uc?id=1ubec-20hOCOOXJgFspuQXHfSH7AjPNcw -O models.zip
unzip models.zip
mkdir -p models
mv hero* models
rm models.zip
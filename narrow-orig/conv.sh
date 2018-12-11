#!/bin/bash

for i in *.jpg;do
	convert -resize 900x200! "$i" "conv-$i";
done

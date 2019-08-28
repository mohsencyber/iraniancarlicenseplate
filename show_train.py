from  PIL import Image , ImageDraw

"""
<images>
  <image file='car01.jpg'>
    <box top='297' left='361' width='217' height='41'/>
  </image>
  <image file='car04.jpg'>
    <box top='187' left='196' width='129' height='12'/>
  </image>
  <image file='car05.jpg'>
    <box top='368' left='881' width='210' height='31'/>
    <box top='1771' left='934' width='126' height='21'/>
    <box top='1409' left='980' width='76' height='12'/>
  </image>
  <image file='car11.jpg'>
    <box top='820' left='625' width='102' height='20'/>
    <box top='219' left='675' width='122' height='27'/>
  </image>
</images>

<image file='2008_001009.jpg'>
    <box top='79' left='145' width='76' height='76'/>
    <box top='214' left='125' width='90' height='91'/>
  </image>
"""

img = Image.open("car_plates/car01.jpg")

imagefinal=ImageDraw.Draw(img)
imagefinal.rectangle(((297,361),(297+217,361+41)),outline="black")

img.show()

img = Image.open("car_plates/car04.jpg")

imagefinal=ImageDraw.Draw(img)
imagefinal.rectangle(((187,186),(187+129,186+42)),outline="black")

img.show()
img = Image.open("car_plates/car05.jpg")

imagefinal=ImageDraw.Draw(img)
imagefinal.rectangle(((368,881),(368+210,881+31)),outline="black")
imagefinal.rectangle(((1771,934),(1771+126,934+21)),outline="black")
imagefinal.rectangle(((1409,980),(1409+76,980+12)),outline="black")

img.show()
img = Image.open("car_plates/car11.jpg")

imagefinal=ImageDraw.Draw(img)
imagefinal.rectangle(((820,625),(820+102,625+20)),outline="black")
imagefinal.rectangle(((219,675),(219+122,675+27)),outline="black")

img.show()

img = Image.open("car_plates/2008_001009.jpg")

imagefinal=ImageDraw.Draw(img)
imagefinal.rectangle(((145,79),(145+76,79+76)),outline="black")
imagefinal.rectangle(((125,214),(125+90,214+91)),outline="black")

img.show()



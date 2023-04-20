with open('assets/style.css', 'rt') as f:
    a = f.read()
import re
c = re.compile('url\(([^)]+)')
lst = c.findall(a)


from urllib import request
from pathlib import Path
for i in lst:
    b = 'https://stackedit.io'
    u = b + i
    
    request.urlretrieve(u, f'assets/static/fonts/{Path(u).name}')

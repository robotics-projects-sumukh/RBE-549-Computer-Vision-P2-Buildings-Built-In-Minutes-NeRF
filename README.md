## Setup for NeRF

1. Go to Phase2 Directory
```bash
cd Phase2
```

2. a) Steps to run training using **Lego Dataset** Approach:
```bash
python3 Wrapper.py --mode=train --data_name=lego
```
2. b) Steps to run training using **Ship Dataset** Approach:
```bash
python3 Wrapper.py --mode=train --data_name=ship
```
2. c) Steps to run training using **Custom Dataset** Approach:
```bash
python3 Wrapper.py --mode=train --data_name=spidey
```

3. a) Steps to run testing using **Lego Dataset** Approach:
```bash
python3 Wrapper.py --mode=test --data_name=lego
```
3. b) Steps to run testing using **Ship Dataset** Approach:
```bash
python3 Wrapper.py --mode=test --data_name=ship
```
3. c) Steps to run testing using **Custom Dataset** Approach:
```bash
python3 Wrapper.py --mode=test --data_name=spidey
```

*Note:* This will generate **image** folder where the results will be saved.

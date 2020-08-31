# Linear controller for quadrotor

Code in this repo was developed based on https://github.com/juanmed/quadrotor_sim/blob/master/3D_Quadrotor/3D_control_with_body_drag.py.

The dynamics decribed in Eq. (2.22) from [this thesis](https://www.kth.se/polopoly_fs/1.588039.1550155544!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf) was adopted.
In order to make use of LQR, the nonlinear dynamics is first linearized around the equilibrium (phi = 0 and theta = 0). We refer the interested readers to this [paper](https://ieeexplore.ieee.org/document/6417914) for details about linearization. Then, LQR is used to calculate a linear controller. The linear controller is then plugged in. The resulting closed-loop linear and nonlinear systems are evaluated either to follow some waypoints or to track a randomly generated trajectory. Finally, the realized trajectories and tracking error for both systems are plotted.

### Requirements
```bash
pip install scipy matplotlib
```

### Usage
```bash
# follow waypoints (-w x1 y1 z1 -w x2 y2 z2 -w x3 y3 z3 -w ... -T total simulation time)
python 3D_quadrotor.py -T 20 -w 1 1 1 -w 1 1 2 -w 0 0 0
# track a randomly genearted trajectory
python 3D_quadrotor.py --seed 0
```

![Follow waypoints](/images/follow_waypoints.png)

# MASAGA
## Description

Official code for the paper ["MASAGA: A Linearly-Convergent Stochastic First-Order Method for Optimization on Manifolds"](http://www.ecmlpkdd2018.org/wp-content/uploads/2018/09/617.pdf). We obtained the projection definitions from ["manopt"](https://www.manopt.org/manifold_documentation_sphere.html).


## Requirements

- Pytorch version 0.4 or higher.

## Running the experiments

### Synthetic data

```
python main.py -e synthetic_L -m train -r 1
```

### Ocean data

```
python main.py -e ocean_L -m train -r 1
```

### Mnist data

```
python main.py -e mnist_L -m train -r 1
```



## Citation 
If you find the code useful for your research, please cite:

```bibtex
@article{babanezhadmasaga,
  title={MASAGA: A Linearly-Convergent Stochastic First-Order Method for Optimization on Manifolds},
  author={Babanezhad, Reza and Laradji, Issam H and Shafaei, Alireza and Schmidt, Mark},
  journal={ECML},
  year={2018}
}
```

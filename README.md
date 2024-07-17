# PyResolveMetrics

This is a standards compliant ([PEP-544](https://peps.python.org/pep-0544/)) and
efficient library for computing entity resolution metrics.
The library currently supports two ways of modelling entity resolution:

* [Fellegi-Sunter](https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501049)
* [algebraic](https://www.igi-global.com/chapter/information-quality-management/23022)

## Set up

This library uses [Poetry](https://python-poetry.org) to manage dependencies and
`make` for build management. Please make sure these tools are installed.

## Run tests

Run the unit tests for this project using `make`.

```shell
$ make test
```

## Usage sample

Sample code that's informative about the library's capabilities
[is available](./sample/sample.py).
Check it out to figure out how the library works.
The [unit tests](./tests/) can also be used as documentation. 

## Paper Citation

This paper was published in the  Proceedings of the 16th International
Conference on Computer Supported Education.

### in Harvard Style
```
Olar A. and Dioşan L. (2024). PyResolveMetrics: A Standards-Compliant and Efficient Approach to Entity Resolution Metrics. In Proceedings of the 16th International Conference on Computer Supported Education - Volume 1: CSEDU; ISBN 978-989-758-697-2, SciTePress, pages 257-263. DOI: 10.5220/0012546300003693
```

### in Bibtex Style
```bibtex
@conference{csedu24,
author={Andrei Olar and Laura Dioşan},
title={PyResolveMetrics: A Standards-Compliant and Efficient Approach to Entity Resolution Metrics},
booktitle={Proceedings of the 16th International Conference on Computer Supported Education - Volume 1: CSEDU},
year={2024},
pages={257-263},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0012546300003693},
isbn={978-989-758-697-2},
}
```

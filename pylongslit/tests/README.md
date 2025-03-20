**PyLongslit test suite**

The data is located at https://github.com/KostasValeckas/PyLongslit_dev . It will be downloaded automatically during 
first time you execute the tests. 

If you already have this data locally (e.x. from following the tutorial), you can re-use it instead. Just update the directory pathes in the configuration files in this directory (see the docs on configuration files at: https://kostasvaleckas.github.io/PyLongslit/).

The test suite is made to perform an automated pipeline blind-run on the two example datasets. This exercises the majority of the code in the pipeline, and ensures all I/O works. It can be useful to run when you are developing the software, as it will tell you if any changes at one place in the pipeline has broken something another place. It will not 
produce correct scientific results, as the interactive part 
is ignored.

However, for a firm test, you have to run the full pipeline manually on the two datasets, and inspect the QA plots manually. 
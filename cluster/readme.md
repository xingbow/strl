### clustering

- For the clustering, most of the codes are run in *jupyter noterbook*

  Basic dataset is provided as sample to run the program.

- for *bike_cluster_diy.ipynb*:

  following work is done sequentially:

   - data preprocessing
   - community generation
   - region generation
   - provide input for *repo-demo.py*

 - for *bike_cluster.ipynb*:

    - Reimplement the clustering in ***Dynamic Bike Reposition: A Spatio-Temporal Reinforcement Learning Approach***

 - for *cluster_vis.ipynb*:

    - Visualization of clustering on Map
    - provide input for demo of reposition *repo-demo.py*

 - for *repo-demo.py*:

    - visualize the reposition route of the trikes. **Reference**: examples of [*geoplotlib*](https://github.com/andrea-cuttone/geoplotlib/tree/master/examples)

      ```python
      python repo-demo.py
      ```

### Requirements

- The code is run under the version of software package is listed below:

```python
pandas 0.23.0
numpy 1.14.3
pyproj 1.9.5.1
scikit-learn 0.20.0 
matplotlib 2.2.2
geoplotlib 0.3.2
plotly 3.4.1
```


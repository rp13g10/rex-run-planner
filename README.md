# Rex Run Planner

This repo forms part of an ongoing project, which aims to generate circular running routes which are optimised for elevation gain/loss. The intention is to generate superlative routes, which result in elevation gains beyond what you'll typically get out of Strava/Komoot. Detailed usage instructions are not provided at this stage, as the entire project is still going through regular refactors as I refine the overall architecture.

# Future Developments

* Elevation gains are currently over-estimated. It is believed that this is due to artifacts in the LIDAR data, but further investigation is required.
    * A basic web-app is being built out to provided the visualization tools required to get some intuition around what's actually going on.
* The existing route-finding algorithm needs refining. Further performance improvements should be possible, and the user should be able to customise the output type beyond just selecting the route & terrain.
* The data ingestion process (handled in the relevation repo) is slated for several improvements.
    * The datastax bulk loader for cassandra needs investigating as it may yield significantly faster data loads
    * The whole stack needs moving across to k8s for scaling/balancing, with airflow in place to trigger ingestion as new files land.
* As the size of available data grows, partitioning strategies need to be examined for the parquet dataset which drives the app. It is not currently clear whether graphframes will natively support filtering of partitioned datasets, or whether a custom implementation will be required. 
* Heavier graph operations such as distance to start tagging need moving across to graphframes to take better advantage of the parallelism provided by pySpark.
* The backend process for the webapp & accompanying stack needs setting up in k8s, in particular the spark driver needs to be separated out.

# Notes

Selector / app currently not working, need to set them up to work with new output format once refinement has been updated to generate partitioned node and edge tables
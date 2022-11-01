# Road Geometry tool

By z.yinglong, z.yinglong@bupt.edu.cn

## Background info of lanes

This tool is build base on the original 2-d labeling logic of this software.

The major content of the lane dataset is the x and y location of the keypoints in the lane.

## label results

This label result is formulated in the Curvelanes format <https://github.com/SoulmateB/CurveLanes>  
These label could be reformed as tusimples format with this tool <https://github.com/pandamax/Parse_Curvelanes/blob/master/curvelanes2tusimples.py>

The result is saved in a json file and in this style:

```python
label:
{
  "Lines":[
    # A lane marking
    [
      # The x, y coordinates for key points of a lane marking that has at least two key points.
      {
        "y":"1439.0",
        "x":"2079.41"
      },
      {
        "y":"1438.08",
        "x":"2078.19"
      },
      ...
    ]
    ...
  ]
}
```

## Dataset Contents

The lane detection dataset labeling content includes different lane types, lane color and many other types:  <https://zhuanlan.zhihu.com/p/406789626>.

And the lane label is related to different road conditions, the standard documentation related to this is: <https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection>

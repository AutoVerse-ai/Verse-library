# Troubleshooting Verse

This is a guide to troubleshooting the most common issues in using Verse.


## Unsuported Operations

Even though the decision logic for mode transitions is written in Python, it is not run in the traditional sense. Verse takes the decision logic and parses it for analysis.

Verse's parser only supports the following:
- if statements
- arithmetic operators (+,-,/,*)
- logical operators (and, or , not)
- comparison operators (<=, <, ==, ! =, >, >=)
- functions
- assertions (for defining safety)
- "any" function
- "all" function


This means that the following are *not* supported (don't even try it):
- print statements
- numpy functions
- else/elif statements
- loops (except any and all statements)


If something is not supported, the parser will usually throw a "Not Supported" error message


## Resolving Infinite Loops

Take this decision logic snippet:

```python

if ego.craft_mode == CraftMode.Normal:
    next.craft_mode == CraftMode.Up

if ego.craft_mode == CraftMode.Normal:
    next.craft_mode == CraftMode.Down

```
We can see that the transition condition is the exact same in both if statements. In this case, both the "up" and "down" branches will both be ran by Verse.

<br/><br/>
Now take, for example, this logic:

```python

if ego.craft_mode == CraftMode.Normal:
    next.craft_mode == CraftMode.Up

if ego.craft_mode == CraftMode.Up:
    next.craft_mode == CraftMode.Normal

```


In this example, we can clearly see a cycle. The mode will transition to "Up", then immediately transition to "Normal", then "Up" again. Avoid situations where a mode transition can occur too soon.
<br/><br/>

To fix this problem, make the transition condition more specific:


```python

if ego.craft_mode == CraftMode.Normal and ego.z <= 50:
    next.craft_mode == CraftMode.Up

if ego.craft_mode == CraftMode.Up and ego.z > 50:
    next.craft_mode == CraftMode.Normal

```

In this scenario, let's assume that z is a state variable that is increasing in the "Up" tactical mode. Now, the transition condition for the first branch (z <= 50) does not overlap with the second branch (z > 50). There is no longer any chance of a cycle occuring.


## Other Issues

### The lower bounds in the initial conditions should always be lower than the upper bounds.


```python

scenario.set_init(
        [
            [[10, 20, 15], [15, 15, 20]],
        ],
       ...
    )
```

We can see that in the second state variable, the lower bound (20) is higher than the upper bound (15). 


### In the decision logic, always modify a copy of "ego" and not "ego" itself. 

There should be a

```python

next =  copy.deepcopy(ego)
```
at the beginning of each decision logic. "next" should be modified instead of "ego".

You also may not modify "others"

### Each agent id must be unique.

If "car_1" is already defined,
```python

car1 = new_agent("car_1", file_name=input_code_name)
```

Do not define it again

```python

car2 = new_agent("car_1", file_name=input_code_name)
```




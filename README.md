# Ragamuffin.jl
**R**ecursive
**AG**gregation
**A**ccelerator

This is an experimental framework for running large, unpredictable recursion problems on a GPU. This is tailored to problems where:
1. Recursive branching pattern depends on input (otherwise one could just invoke a sequence of kernels which worked for all input values)
2. Each step has relatively heavy math (so that runtime isn't dominated by data movement)
3. Steps require heap memory accesses (counter example: mathematical recurrences which are simple functions of numbers)
4. The affect of each branch on the output is independent of other branches (aggregation)

This was motivated by an interest in ray tracing and a need to separate GPU logic from interesting optics. I'm playing with different physical behaviors and ways of approximating continua of rays as polynomials.


# Examples

In `examples` you can check out three toy calculations. None of them are more performant than the simple CPU implementations used to verify correctness. This is because none of these problems meet all four criteria above.

# Model
Every recursive problem can be modeled as a process.

Here is the recursive definition for calculating the stopping time for the Collatz Sequence:
```
function collatz_length(x)
    if x == 1
        return 0
    elseif x % 2 == 0
        return 1 + collatz_length(x / 2)
    else
        return 1 + collatz_length(3 * x + 1)
    end
end
```
The same algorithm can be visualized as a process:
![Process Diagram](https://github.com/xaellison/Ragamuffin.jl/blob/master/collatz_process.png)

Ragamuffin accelerates these kinds of calculations by parallelizing each step for several values.  It takes many input values, follows the process, and stores the result of each input value in an output array. It starts at `root`. Batches of values are stored in 'frames'. A frame consists of many 'cells' which hold the value and the output index.

A step can...
1. straight evaluate a function (eg: `increment` above)
2. evaluate a function of the value and each auxiliary variable in some global array. It can **reduce** over those auxiliary variable for each value (this is like `select x MIN(y) ... group by y` in pseudo-SQL).
3. **branch** on each value by calculating some condition. Those meeting the condition are put in a frame for one destination and those which failed are put in a frame for another destination.
4. **bifurcate**: values are copied into frames for several other steps (one-to-many).

# Execution
Every step in a process has a stack of frames. Depending on the type of step, values are calculated and moved into one or more frames which can go to different downstream steps. The program executes as long as some stack still has frames in it.

It is possible that only a small fraction of values will fail/meet a condition, which can lead to frames with very few values. It would be inefficient to invoke kernels on these. So each step has a separate stack of 'partial' frames. Once there are enough values in these partial frames, they are collapsed into a single frame and put on that step's main stack.

If there are only values to run in partial frames, then the collapse step is bypassed. Such executions happen towards the end of calculations.

A pool of frames is maintained to avoid reallocation.
# Usage
I'm still working out the underlying mechanics, so I haven't polished the interface yet. Eventually there will be convenience macros, but for now see the examples.

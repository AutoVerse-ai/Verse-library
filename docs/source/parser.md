# Parser

## Goals

The primary goal of the parser is to obtain single expressions for different parts of the system, so that it's easier for the backend system to do analysis. Since we use a SAT solver (z3) in the backend for checking if a guard in the controller is activated, a single expression would allow us to convert the expression directly into a z3 expression and the satisfiability is checked once.

## Working Principle

The parser achieves this by using a simple AST walking interpreter on the input file, with a list of scoped environments to keep track of the values of the variables. The only difference to normal interpreters being that the values of the variables are the ASTs themselves, i.e. no evaluation is done on the ASTs. When the variables are used, the ASTs they contain are substituted into the larger AST they are in. Functions and lambdas work essentially the same way, with the body of the function/lambda substituted with the arguments' ASTs (functions and lambdas actually use the same value augmented to the Python AST i.e. `Lambda`)

Since the values can only be some form of Python AST (or custom values, as explained below), no builtin functions/variables are supported other than the ones mentioned in [Reductions](#Reductions). Imported modules are also not processed. They are treated as `Hole`s in the environments, i.e. valid defined symbols but the parser knows nothing more about them. Functions that are called as a member of a module or of a variable are preserved and not expanded, as they may be supported by the backend. The use of any other functions will cause an error. 

In order to simplify backend processing, there are 2 new values introduced into the AST (in addition to lambdas), `CondVal`s and `Reduction`s.

### Conditional Values

`CondVal`, or conditional values, are used as the primary analysis component for the backend. They store a list of all the possible values a variable can take, as well as the conditions that needs to be met for the values to be taken (i.e. a list of `CondValCase`s). In each case, the list of conditions are implicitly `and`ed together.

`CondVal`s are constructed when a variable is assigned in a if statement somewhere in the code. When a variable is assigned multiple times in different if statements (or different branches of them), the value will be "merged" together using the test condition. Simply, when a merge happens, the test condition is appended to the list of conditions for each value in the `CondVal`, and if the value is assigned in an else branch, the test is inverted.  

Due to how the execution uncertainty is checked in the backend, the parser also doesn't follow the usual semantics of how `if` statements work. Consider this snippet:

```python
a = 42
if test:
    a = 3
```

The parser will actually report that `a` could be `42` whenever (no condition needed to trigger) and `3` when `test` is satisfied. In a normal python execution, `a` will only be `42` if `test` is not satisfied (unless the evaluation of `test` causes an exception).

### <a name="Reductions"></a>Reductions

Due to the need to support multiple agents, it's necessary to have support for some form of reductions, since the control output will only be one "thing". Arbitrary loops are hard to support, since they can have arbitrary control flow that are hard to analyze. Instead, we support builtin reduction functions (currently only `all` and `any`) called on generator comprehension expressions. These are much better formed and easier to process. When the parser encounters a call like `all(test(o) for o in other_states)` it'll convert that into a `Reduction` value, which will then be unrolled in reachtube computation with the sensor values, or converted back to a normal Python function call to be evaluated in simulation.

## Limitations

- Very limited function definition support
- Basically no support for imported modules 
- No loops allowed in the code, instead only specific reduction calls are supported
- Unusual if statement semantics
- Only one return statement is supported per function, and it has to be at the end
- State definitions are inferred from the variable type definitions in the class that end in `State`, and the type (discrete/continuous) of the member variable is determined from the name; no method definition is used
- Similar to how states are processed, discrete modes are also class definitions with names ending in `Mode`, and the declaration is assumed to be how `enum.Enum` is usually used; see demos for examples. No method definition is used
- Custom class definitions (other than the states and modes definitions) are not processed

Some of these may be resolved later, others stem from the limitations of some of the analysis methods and backends (e.g. z3)

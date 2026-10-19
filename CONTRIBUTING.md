# Contributing to relearn

Thank you for helping! relearn is used for teaching as much as for building agents,
so code is written to be **read**: by the maintainer, who reviews every change by hand,
and by students meeting reinforcement learning for the first time.

## Ground rules

- **`src/relearn.hpp` stays header-only, with no required dependencies.**
  It must build with nothing but a C++14 compiler, so that it runs on embedded boards,
  drones and robots. Optional features (such as Boost serialization) stay behind a macro.
- **Examples and games may use third-party libraries**, but only in their own optional
  CMake project, off by default. The default build must stay dependency-free.
- **Every change comes with a test**, and every learning test is checked against a known
  answer (an exact solution, or a result from the book), across several random seeds.
- **Keep pull requests small** enough to review in one sitting.
- **AI-generated contributions are welcome**, as long as they have been reviewed and approved by a human.

## Code style

The style below is the one used in `src/relearn.hpp` and the gridworld examples.

### Layout

- Indent with 4 spaces, never tabs.
- Functions, classes, structs and namespaces open their brace on a **new line**.
  `if`, `for`, `while` and `switch` open it on the **same line**, and `else` starts a new line:

  ```cpp
  void walk(const world & w)
  {
      if (blocked) {
          turn();
      }
      else {
          step();
      }
  }
  ```

- Always use braces, even for a single statement, and put one statement on each line.
- Template parameters and long parameter lists go one per line, aligned:

  ```cpp
  template <class state_class,
            class action_class>
  std::deque<link<state_class,action_class>> explore(const world & w,
                                                     std::mt19937 & gen,
                                                     grid start);
  ```

- A `case` label goes on its own line, with its statements and `break` indented below it.
- Line up the `=` of related consecutive assignments when it helps reading.

### Names

- `snake_case` for everything: types, functions, variables and constants.
- Private and protected data members end with an underscore: `trait_`, `policies_`.
- Never use a name containing a double underscore (`__name__`), or starting with an
  underscore and a capital letter: the C++ standard reserves them for the compiler.
- Template parameters describe what they are (`state_trait`, `action_class`),
  or are single capitals in short examples (`S`, `A`).

### Comments

- Every class and struct has a Doxygen block with `@brief` (and `@class` or `@struct`),
  explaining what it is for.
- Every function has a `///` line or a `/** */` block, using `@param`, `@return`,
  `@warning`, `@see` and `@note` where they help.
- Inside functions, a short plain-English `//` comment before each step says **what the
  step does and why**, so a student can follow the algorithm without the book open.
- When code implements a formula, write the formula in the comment, in the notation of
  Sutton & Barto, and cite the section.

### Code

- Write it simply: prefer a named helper function to a long lambda, and a few clear lines
  to one clever line.
- Print with `std::cout` and `std::endl`, using `<iomanip>` for number formatting.
- Use `auto` when the type is obvious from the right-hand side, and `using` aliases for
  long types.
- Headers use include guards named after the path: `RELEARN_GAMES_GRID_HPP`.
- Example programs return `EXIT_SUCCESS` only if the agent learnt what it should,
  so that CI notices when an example stops working.

## Building and testing

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=On
make
ctest --output-on-failure
```

Run the example programs you changed as well: each one exits with a failure code when
its agent does not learn.

![RELEARN](https://github.com/alexge233/relearn/blob/master/images/relearn.png?raw=true)

[![check](https://github.com/alexge233/relearn/actions/workflows/check.yaml/badge.svg)](https://github.com/alexge233/relearn/actions/workflows/check.yaml)

[English](README.md) | **Español**

__nota__: esta es la traducción al español del [README](README.md). La documentación de cada algoritmo
(en `docs/`) y la guía para contribuir están en inglés. Si las dos versiones no coinciden, la inglesa es la de referencia.

Una biblioteca de plantillas de aprendizaje por refuerzo para C++14, *header-only* (solo headers).
Mínima y sencilla de usar, para escenarios muy diversos.
Está basada en el libro de Sutton y Barto [1] e implementa *parte* de lo que se describe en él.

Actualmente la biblioteca implementa los siguientes algoritmos, cada uno explicado en su propio documento
(con las fórmulas, el pseudocódigo y cómo usarlo):

- **Q-Learning** (`q_learning`) para sistemas deterministas, y también para sistemas no deterministas (`q_probabilistic`),
  ver [docs/q_learning.md](docs/q_learning.md)
- **SARSA** (`sarsa`) y **Expected SARSA** (`expected_sarsa`), que aprenden el valor de la política que el agente
  sigue de verdad, exploración incluida, ver [docs/sarsa.md](docs/sarsa.md)
- **Double Q-Learning** (`double_q_learning`), que guarda dos estimaciones, de modo que la que elige la mejor acción siguiente
  no es la que la valora, y así no se deja engañar por valores con ruido, ver [docs/double_q_learning.md](docs/double_q_learning.md)
- **Dyna-Q** y **Dyna-Q+** (`dyna_q`), que recuerdan todos los movimientos que hacen y los repasan mentalmente (planificación)
  para aprender mucho más de la misma experiencia, mientras que Dyna-Q+ además vuelve a comprobar los movimientos que
  no hace desde hace tiempo, por si el mundo ha cambiado, ver [docs/dyna.md](docs/dyna.md)
- **control Monte Carlo** (`monte_carlo` y `off_policy_monte_carlo`), que espera a que termine cada episodio
  y aprende de las recompensas que realmente siguieron, con inicios exploratorios, on-policy (ε-Greedy), u off-policy
  con muestreo por importancia, ver [docs/monte_carlo.md](docs/monte_carlo.md)
- predicción **TD(0)** y **TD de n pasos** (`n_step_td`), y **SARSA de n pasos** (`n_step_sarsa`), que aprenden de
  las n recompensas siguientes y después del valor del estado al que llegan, en cualquier punto entre el TD de un paso
  y Monte Carlo, ver [docs/n_step.md](docs/n_step.md)
- **TD(λ)** (`td_lambda`), **SARSA(λ)** (`sarsa_lambda`) y **Q(λ) de Watkins** (`watkins_q_lambda`), cuyas
  trazas de elegibilidad reparten cada error TD entre los movimientos que llevaron a él, acumulativas o de reemplazo,
  ver [docs/eligibility_traces.md](docs/eligibility_traces.md)
- **REINFORCE**, **REINFORCE con línea base** y **Actor-Critic de un paso** (`reinforce`, `reinforce_with_baseline`
  y `actor_critic`), que aprenden la propia política, como un softmax sobre las preferencias de cada acción (`softmax_policy`),
  de modo que pueda seguir siendo estocástica cuando eso sea lo mejor, ver [docs/policy_gradient.md](docs/policy_gradient.md)
- **R-Learning** (`r_learning`) y **Sarsa diferencial** (`differential_sarsa`), para tareas que no terminan nunca:
  aprenden la recompensa media por paso, y cuánto mejor que la media es cada acción, después de cada paso,
  ver [docs/r_learning.md](docs/r_learning.md)
- **Prioritized sweeping** (`prioritized_sweeping`), que planifica como Dyna-Q, pero repasa primero los movimientos cuyos valores
  cambiarían más, yendo hacia atrás desde cada cambio, ver [docs/prioritized_sweeping.md](docs/prioritized_sweeping.md)
- **Programación dinámica** (`dynamic_programming`, con un modelo `finite_mdp`): evaluación de políticas, iteración de políticas
  e iteración de valores, que no necesitan jugar en absoluto cuando se conocen exactamente las reglas del mundo,
  ver [docs/dynamic_programming.md](docs/dynamic_programming.md)
- **UCB** (`ucb`) y el **bandido de gradiente** (`gradient_bandit`), para el bandido de k brazos: UCB explora los brazos
  de los que está menos seguro, mientras que el bandido de gradiente aprende una preferencia por cada brazo, y elige los
  brazos con su softmax, ver [docs/bandits.md](docs/bandits.md)
- **schedules** (pautas de decaimiento) para la tasa de aprendizaje y la tasa de exploración, de modo que cada una pueda *decaer*
  a medida que el agente aprende (`polynomial_decay`, `exponential_decay` y `linear_decay`), ver [docs/schedules.md](docs/schedules.md).
  Ten en cuenta que `q_learning`, `sarsa`, `expected_sarsa`, `double_q_learning` y `n_step_sarsa` pueden hacer decaer, cada uno,
  la tasa de aprendizaje de cada par estado/acción por separado

También incluye una colección de juegos clásicos para aprender, descritos más abajo en [Juegos](#juegos).

## Implementación en C++

La clase de plantilla `policy` guarda los valores aprendidos de tus estados y acciones.
El header ofrece un par de clases envoltorio de plantilla, `state` y `action`, que envuelven tus propias
estructuras de estado y de acción (el `state_trait` y el `action_trait`). Pueden ser cualquier cosa, siempre que:
- tengan un `operator==`
- se les pueda calcular un hash, con una especialización de `std::hash`

Asegúrate de que sean únicos: dos estados distintos cuyos traits sean iguales se tratan como uno solo.
El archivo `examples/gridworld_header.hpp` muestra cómo escribirlos.

También puedes crear tus propias clases `state` y `action` y usarlas en lugar de los envoltorios,
pero deben cumplir los siguientes criterios:
- tener un `operator==`
- tener una especialización de `relearn::hasher`
- los estados deben exponer su recompensa con un método `reward()`

### Proceso de decisión de Markov

Un MDP (*Markov Decision Process*) es una secuencia de estados s<sub>t</sub> enlazados por acciones a<sub>t</sub>.
En el centro de `relearn` está la `policy`, que usa los MDP y los **mapea** con las recompensas `R` recibidas a lo largo de ellos.
Aunque puede haber varios episodios, y un episodio puede recorrerse muchas veces, actualizarse, evaluarse
y volver a experimentarse, el proceso en sí se hace en la clase `policy`. Abajo se muestra un esquema de un MDP.

![Markov Decision Process](https://github.com/alexge233/relearn/blob/master/images/mdp.png?raw=true)

Los parámetros de plantilla `state_trait` y `action_trait` sirven para describir qué es un estado y qué es una acción,
mientras que los envoltorios solo se usan para indexar, calcular el hash y calcular la política.

El uso básico consiste en crear episodios, llamados `markov_chain`, que rellenas como mejor te parezca,
dando a cada paso la recompensa que obtuvo (cualquier valor: positivo, negativo o cero).
Ten en cuenta que cada recompensa pertenece al paso que la obtuvo. Échale un vistazo a la
[convención de las recompensas](docs/q_learning.md#reward-convention) (en inglés).

Por defecto, un `markov_chain` es un `std::deque<relearn::link<state_class,action_class>>`,
donde `relearn::link` es la estructura que encapsula una observación estado/acción.

Puedes cambiar el contenedor o la clase del enlace si quieres, pero los algoritmos de aprendizaje necesitan que:

- el episodio tenga un `operator[]` que reciba el índice de un enlace, y un método `size()`
- cada enlace tenga los miembros `state` y `action`

Básicamente `std::deque` y `std::vector` funcionan tal cual (`std::list` no, porque no tiene `operator[]`).
La biblioteca usa por defecto `double` para guardar y actualizar los valores Q,
pero puedes cambiarlo si lo necesitas.

La clase que más nos interesa es `policy`, que usa tus estados y acciones,
y los guarda en un mapa de estados, donde cada estado (clave) tiene como valor un mapa (de acción a valor):

```
.
├── s_t
|   ├── a_t: 0.03
|   └── a_t: -0.05
├── s_t
|   ├── a_t: -0.001
|   └── a_t: 0.9
```

Los objetos `policy` son tuyos y los controlas tú, e incluso puedes usar varios a la vez, pero ten en cuenta
que no están protegidos para el acceso desde varios hilos (tienes que hacerlo tú a mano).
Con cualquiera de los algoritmos de aprendizaje puedes asignar o actualizar los valores de la política recorriendo tus episodios.

# Dependencias

¡No hay **ninguna dependencia externa**!
Eso sí, tu compilador **tiene que ser compatible con C++14**, así que necesitarás:

- GCC 5 o superior
- Clang 3.4 o superior

Son las primeras versiones con compatibilidad completa con C++14. La CI compila con GCC en Ubuntu, y con Apple Clang en macOS.

__nota__: ¡no lo he probado en Windows!

__nota__: para compilar los ejemplos y los tests se necesita CMake 3.5 o superior, y los tests usan
[Catch2](https://github.com/catchorg/Catch2) v2, cuyo único header descarga CMake.

__nota__: la serialización es opcional, y necesita Boost (ver [activar la serialización](#activar-la-serialización) más abajo).

# Compilación

¡No hay nada que compilar! Es una biblioteca *header-only*, solo necesitas la carpeta `src` e incluir el header **relearn.hpp**.
**Sin embargo**, puedes compilar los ejemplos para ver cómo se usa la biblioteca.

`relearn.hpp` simplemente incluye las partes de la biblioteca, que están en `src/relearn/`, un header para cada una:

| header | qué contiene |
|---|---|
| `relearn/core.hpp` | `state`, `action`, el `link` de un episodio, y la `policy` |
| `relearn/schedules.hpp` | `polynomial_decay`, `exponential_decay`, `linear_decay`, y el `visit_counter` |
| `relearn/q_learning.hpp` | `q_learning` y `q_probabilistic` |
| `relearn/sarsa.hpp` | `sarsa` y `expected_sarsa` |
| `relearn/double_q_learning.hpp` | `double_q_learning` |
| `relearn/dyna.hpp` | `dyna_q` |
| `relearn/monte_carlo.hpp` | `monte_carlo` y `off_policy_monte_carlo` |
| `relearn/n_step.hpp` | `n_step_sarsa`, y `n_step_td` (TD(0) cuando n = 1) |
| `relearn/eligibility_traces.hpp` | `sarsa_lambda`, `td_lambda` y `watkins_q_lambda`, y sus `eligibility_traces` |
| `relearn/policy_gradient.hpp` | `softmax_policy`, `reinforce`, `reinforce_with_baseline` y `actor_critic` |
| `relearn/r_learning.hpp` | `r_learning` y `differential_sarsa` |
| `relearn/prioritized_sweeping.hpp` | `prioritized_sweeping` |
| `relearn/dynamic_programming.hpp` | `finite_mdp` y `dynamic_programming` |
| `relearn/bandits.hpp` | `ucb` y `gradient_bandit` |

Cada uno incluye todo lo que necesita, así que si solo usas un algoritmo de aprendizaje (p. ej., en una placa pequeña, donde
cuenta cada byte y cada segundo de compilación) puedes incluir solo ese, `#include "relearn/q_learning.hpp"`.

Para eso, simplemente:

```bash
mkdir build
cd build
cmake ..
make
```

Esto compila todos los ejemplos, como `ex_<nombre>` en la carpeta `build`.

### compilar los tests

```bash
cmake .. -DBUILD_TESTS=On
make
ctest
```

### activar la serialización

Para activar la serialización necesitas boost y boost-serialization.
Consulta la documentación de tu distribución para instalarlos.
Para activarla, pasa una opción a cmake:

```bash
cmake .. -DUSING_BOOST_SERIALIZATION=On
```

Por ejemplo, si quieres ejecutar los tests **con** serialización:

```bash
cmake .. -DUSING_BOOST_SERIALIZATION=On -DBUILD_TESTS=On
```

También puedes usar la serialización en tu **propio** proyecto, si quieres guardar y cargar
políticas, estados o acciones: simplemente define `USING_BOOST_SERIALIZATION` (p. ej., `-DUSING_BOOST_SERIALIZATION`)
antes de incluir `relearn.hpp`, y usa `find_package` de CMake para encontrar, incluir y enlazar `boost_serialization`.
Ten en cuenta que el `state_trait` (es decir, el **descriptor** de tu estado) y el
`action_trait` (es decir, el **descriptor** de tu acción) **también tienen que ser serializables**.
Para saber cómo hacerlo, [échale un vistazo a este tutorial](http://www.boost.org/doc/libs/1_64_0/libs/serialization/doc/tutorial.html);
si no se cumple esta condición, tendrás errores de compilación.

Gracias a la flexibilidad de la serialización de boost, puedes guardar y cargar archivos binarios, de texto o xml.
Las versiones más recientes de boost admiten punteros inteligentes, así que aunque tus descriptores sean
`std::shared_ptr` o `std::unique_ptr` puedes guardarlos y cargarlos igualmente.

# Ejemplos

Hay una carpeta `examples` con dos ejemplos de "Gridworld":
- uno offline, que aprende de una exploración al azar: `examples/gridworld_offline.cpp`, compilado como `ex_gridworld_offline`
- uno online, que aprende después de cada episodio: `examples/gridworld_online.cpp`, compilado como `ex_gridworld_online`

y una colección de juegos clásicos, descritos más abajo en [Juegos](#juegos).

## uso básico

La forma básica de usar la biblioteca es la siguiente:

1. crea una clase **state**, o usa una clase, estructura o POD que ya tengas, que describa (en el sentido *markoviano*) tu estado
2. crea una clase **action**, o usa una clase, estructura o POD que ya tengas, que describa la acción
3. crea un *episodio*, que por defecto es un `std::deque<relearn::link<state,action>>`, y rellénalo con los pasos dados y la recompensa que obtuvo cada uno

A partir de aquí, según uses un algoritmo o enfoque **online** u **offline**, tienes estas opciones:

4. seguir creando episodios y, cuando termines, entrenar la política con todos ellos,
**o**
5. cada vez que crees un episodio, entrenar tu política con él.

Esa elección es tuya, y casi siempre depende del dominio, del sistema o del problema que quieras resolver.
Por esta razón no hay ninguna implementación de `on_policy`, `off_policy` o `e_greedy`:
son algoritmos muy sencillos, y dependen de cada aplicación.
Los juegos muestran una forma de escribirlos (`games::greedy` y `games::epsilon_greedy` en `examples/games/game.hpp`).
UCB y el bandido de gradiente, en cambio, son algoritmos completos para bandidos, así que están en la biblioteca
(échale un vistazo a [docs/bandits.md](docs/bandits.md)).

Échale un vistazo a los ejemplos de `gridworld`, que muestran dos formas distintas de hacer la misma tarea.
El ejemplo de `blackjack` es diferente: el blackjack es un juego de *azar*, donde la misma decisión puede ganar o perder
según las cartas que salgan después, así que usa `q_probabilistic`, que aprende la recompensa esperada y la
probabilidad de cada transición a partir de las partidas que juega.
El ejemplo `blackjack_monte_carlo` aprende el mismo juego con Monte Carlo, de tres formas distintas.

## Gridworld

![gridworld image](https://github.com/alexge233/relearn/blob/master/images/gridworld.png?raw=true)

Lo más sencillo que hay entre los problemas de juguete de mundos de bloques o cuadrículas: nuestro agente vive en un gridworld de 10x10,
rodeado de bloques a los que no puede moverse (en negro).
El agente empieza en el azul (x:1,y:8) y el objetivo es el verde (x:1,y:1).
Los bloques rojos son fuego, peligro, una recompensa negativa, y hay un laberinto rudimentario.
Llegar al objetivo da +1, meterse en un bloque de fuego da -1, y los dos terminan el episodio.

Las dos versiones aprenden con Q-Learning, que aprende los valores de las mejores acciones, sean cuales sean las acciones
que se tomaron mientras se exploraba. Se diferencian en cómo exploran. El enfoque offline:

- primero el agente explora al azar hasta encontrar el bloque con la recompensa positiva (+1.0)
- después actualiza sus políticas
- por último sigue la mejor política aprendida

Y el enfoque online:

- el agente explora un episodio
- después actualiza sus políticas
- después lo vuelve a intentar, esta vez siguiendo las políticas que conoce
- solo vuelve a moverse al azar cuando no existe una mejor acción *positiva* (o cuando esta lleva contra una pared)
- todo el proceso se repite hasta descubrir el objetivo.

El gridworld en sí está guardado en un archivo de texto, `examples/gridworld.txt` (puedes cambiarlo si quieres).
El header `examples/gridworld_header.hpp` contiene el código mínimo para mostrar este enfoque por etapas.
Los dos archivos:

- `examples/gridworld_offline.cpp`
- `examples/gridworld_online.cpp`

tienen las distintas versiones de cómo puede resolverse esta tarea.

Una vez cargado el mundo (con la función `populate`), fijamos el inicio en x:1, y:8 y
empezamos la exploración.

### q-learning offline

La exploración offline se ejecuta en un bucle infinito hasta encontrar el bloque con una recompensa **positiva**.
Hasta que eso ocurra, el agente adopta un enfoque *estocástico* (es decir, al azar) y recorre el gridworld.
La función (el parámetro de plantilla `S` es el estado, y `A` la acción):

```cpp
template <typename S, 
          typename A>
std::deque<relearn::link<S,A>> explore(const world & w,
                                       std::mt19937 & gen,
                                       grid start);
```

hace lo siguiente:

- crea un episodio nuevo (es decir, un `relearn::markov_chain`)
- empieza desde el bloque de inicio
- elige una dirección al azar (ver la estructura `rand_direction` para más detalles)
- registra cada paso con la recompensa del bloque al que se movió
- repite esto hasta que (a) encuentre una recompensa negativa (es decir, pise un bloque de fuego), o (b) descubra el bloque objetivo

Cada vez que se llama al método `explore` en el bucle principal de exploración,
se obtiene un episodio, que luego se añade a la lista de episodios experimentados, `std::vector<std::deque<relearn::link<state,action>>> episodes`.

Cuando el agente ha encontrado el bloque objetivo, actualiza la memoria de su política con esos episodios:

```cpp
auto learner = relearn::q_learning<state,action>{0.9, 0.9};
for (int k = 0; k < 200; k++) {
    for (const auto & episode : episodes) {
        learner(episode, policies);
    }
}
```

En este caso, el agente hace doscientas pasadas por cada episodio, porque cada pasada lleva la recompensa
solo un paso más hacia el inicio, y los episodios al azar pueden ser largos. Los parámetros de Q-Learning son:
- alpha (tasa de aprendizaje) 0.9
- gamma (factor de descuento) 0.9

Por último, cuando ha terminado de actualizar las políticas, el agente sigue (una vez) la política de la mejor acción:

```cpp
on_policy(w, policies, start);
```

El resultado es que el agente va directo de x:1, y:8 a x:1, y:1, esquivando los bloques de fuego
y sin recorrer el gridworld innecesariamente.

Este es un escenario __determinista__, porque el agente sabe en todo momento qué acción está tomando
y a __qué__ estado lo llevará esa acción.

### q-learning online

La exploración online es algo diferente, porque el método `explore` hace lo siguiente:

- crea un episodio nuevo
- empieza desde el bloque de inicio
- si existe una buena política (un valor Q mayor que cero, para un movimiento que no lleve contra una pared), la sigue
- si no existe ninguna buena política, toma una acción al azar
- repite hasta encontrar una recompensa, y entonces **entrena** las políticas con el último episodio (diez pasadas)
- un episodio que deambula durante más de 1000 movimientos se corta, y se aprende con `terminated = false`
  (ver [episodios cortados](docs/q_learning.md#episodes-cut-short), en inglés), para que su último movimiento no se aprenda como el final del juego

La diferencia aquí es que la exploración se ve afectada al instante por lo que ya se ha aprendido.
En cambio, el método offline no se ve afectado, y puede repetir las mismas secuencias una y otra vez.
Sin embargo, si la versión online se detiene demasiado pronto, no hay ninguna garantía de que el agente haya aprendido el
camino ideal u óptimo hasta el objetivo: podría ser simplemente un camino *mediocre* o *tonto* que ha descubierto.
Por supuesto, esto también es un problema del método offline, donde puede que la solución no se descubra nunca.

Existen otros enfoques (p. ej., ε-Greedy) en los que el agente puede seguir la política,
pero decidir al azar ignorarla, para intentar descubrir una solución mejor.
Los juegos de abajo usan ε-Greedy.

# Juegos

La carpeta `examples/games` contiene problemas clásicos de aprendizaje por refuerzo, cada uno en su propio header,
y `examples/<juego>.cpp` aprende cada uno (compilado como `ex_<juego>`). La mayoría vienen de Sutton y Barto [1],
mientras que FrozenLake, cart-pole y mountain car siguen las reglas y la física de Gymnasium [2], para que más adelante
se puedan comparar agentes en los dos.

Todos los juegos tienen la misma forma, inspirada en el `reset()`/`step()` de Gymnasium (ver `examples/games/game.hpp`):

```cpp
games::cliff_walking game;
auto s = game.reset();                        // empezar un episodio
auto out = game.step(games::right);           // out.next, out.reward, out.terminated
auto moves = game.actions(s);                 // las acciones permitidas
```

y `games::play` registra un episodio con la convención de recompensas de relearn, listo para un algoritmo de aprendizaje:

```cpp
using types = games::types<games::cliff_walking>;
types::policy memory;
relearn::q_learning<types::state, types::action> learner;
auto r = games::play(game, [&](const auto & s, const auto & legal) {
    return games::epsilon_greedy<games::cliff_walking>(memory, s, legal, 0.1, prng);
}, 1000);                                     // como mucho 1000 movimientos
learner(r.episode, memory, r.terminated);     // un episodio cortado no se aprende como terminado
```

Otra opción es `games::play_online`, que juega un episodio y aprende después de cada paso, como hacen los
algoritmos TD de Sutton y Barto. SARSA lo necesita con una tasa de aprendizaje alta (échale un vistazo a [docs/sarsa.md](docs/sarsa.md)):

```cpp
auto r = games::play_online(game, agent, learner, memory, 1000);
```

Los algoritmos de n pasos necesitan los últimos n pasos después de cada paso, no solo el último, así que dale a `play_online` su n:
`games::play_online(game, agent, learner, memory, 1000, learner.n)` (échale un vistazo a [docs/n_step.md](docs/n_step.md)).
Los algoritmos con trazas de elegibilidad aprenden un episodio entero cada vez, así que usa `games::play` con ellos.

| Juego | Fuente | Se aprende con | Qué muestra |
|---|---|---|---|
| `bandit` | S&B 2.3 | `q_probabilistic` | Exploración frente a explotación, greedy frente a ε-Greedy (Figura 2.2) |
| `bandit_ucb_gradient` | S&B 2.7, 2.8 | `ucb`, `gradient_bandit` | UCB gana más que ε-Greedy (Figura 2.4), y el bandido de gradiente necesita su línea base (Figura 2.5) |
| `blackjack` | S&B 5.1 | `q_probabilistic` | Un juego de azar, donde la estrategia aprendida coincide con la Figura 5.2 |
| `blackjack_monte_carlo` | S&B 5.3, 5.4, 5.7 | `monte_carlo`, `off_policy_monte_carlo` | El mismo juego aprendido de partidas enteras: con inicios exploratorios, on-policy, y off-policy a partir de partidas jugadas al azar |
| `cliff_walking` | S&B 6.6 | `q_learning`, `sarsa`, `expected_sarsa` | On-policy frente a off-policy: Q-Learning aprende el camino por el borde del acantilado, SARSA uno más seguro, y los dos SARSA ganan más mientras exploran |
| `windy_gridworld` | S&B 6.5, Ej. 6.10 | los dos Q-Learning | Viento constante: el camino de 15 movimientos. Viento aleatorio: `q_probabilistic`, y `q_learning` con una tasa de aprendizaje que decae por par, encuentran la mejor política, mientras que `q_learning` con α = 0.5 no |
| `random_walk` | S&B 6.2, 7.1 | `q_probabilistic` | Predicción, con los valores aprendidos junto a los exactos |
| `random_walk_td` | S&B 6.2, 7.1 | `n_step_td`, `monte_carlo` | TD(0) aprende más rápido que Monte Carlo (Ejemplo 6.2), y una n intermedia es la mejor (Figura 7.2) |
| `random_walk_lambda` | S&B 12.1, 12.2 | `td_lambda` | Una λ intermedia es la mejor (Figura 12.6), y TD(λ) offline aprende exactamente lo mismo que el retorno λ |
| `maximization_bias` | S&B 6.7 | `q_learning`, `double_q_learning` | La sobreestimación de Q-Learning, que Double Q-Learning elimina (Figura 6.5) |
| `gamblers_problem` | S&B 4.3 | `q_probabilistic` | Valores que coinciden con la solución exacta |
| `dynamic_programming` | S&B 4.1, 4.3, 4.4 | `dynamic_programming` | El gridworld de 4 x 4 (`small_gridworld`, Figura 4.1) y el problema del jugador (Figura 4.3), resueltos a partir de sus modelos, con evaluación de políticas, iteración de políticas e iteración de valores |
| `dyna_maze` | S&B 8.1 a 8.3 | `dyna_q` | La planificación aprende un laberinto en muchos menos episodios (Figura 8.2), y Dyna-Q+ se adapta a laberintos que cambian (Figuras 8.4, 8.5) |
| `dyna_maze_n_step` | S&B 7.2, 8.1 | `n_step_sarsa` | El primer episodio le enseña a SARSA de n pasos sus últimos n movimientos (Figura 7.4), así que una n mayor aprende el laberinto más rápido |
| `dyna_maze_lambda` | S&B 12.7, 12.10 | `sarsa_lambda`, `watkins_q_lambda` | Con trazas de reemplazo, basta un episodio para volver a encontrar la meta; las trazas acumulativas se pierden en los bucles |
| `short_corridor` | S&B 13.1 | `reinforce`, `reinforce_with_baseline`, `actor_critic` | Casillas que parecen iguales, donde la mejor política es estocástica: REINFORCE la aprende (Figura 13.1), más rápido con línea base (Figura 13.2), mientras que el crítico de Actor-Critic lo despista |
| `access_control` | S&B 1.ª ed. 6.7, 2.ª ed. 10.2 | `r_learning`, `differential_sarsa` | Una tarea que no termina nunca: aceptar o rechazar clientes para ganar lo máximo por paso, frente a la política óptima exacta (Figuras 6.17, 10.5) |
| `dyna_maze_prioritized` | S&B 8.4 | `prioritized_sweeping`, `dyna_q` | En laberintos cada vez más grandes, prioritized sweeping necesita muchas menos actualizaciones que Dyna-Q para encontrar un buen camino (Ejemplo 8.4) |
| `tic_tac_toe` | S&B 1.5 | `q_learning` | Contra un rival al azar o perfecto (minimax) |
| `frozen_lake` | Gymnasium | `q_probabilistic` | Hielo resbaladizo, donde la política aprendida en 4x4 es óptima |
| `cart_pole` | Gymnasium | `q_learning` | Un estado continuo, dividido en casillas (`games::discretised`) |
| `mountain_car` | Gymnasium | `q_learning` | Un estado continuo, explorado con valores iniciales optimistas |

El archivo `tests/games_test.cpp` comprueba las reglas de cada juego, y que los algoritmos llegan a la respuesta conocida:
o bien soluciones exactas (`tests/solvers.hpp`: value iteration, relative value iteration, blackjack exacto, búsqueda en anchura) o bien los
resultados óptimos que da el libro.

# Contribuir

¡Las contribuciones son bienvenidas! Por favor, lee primero [CONTRIBUTING.md](CONTRIBUTING.md) (en inglés),
que describe el estilo del código y cómo se prueban los cambios.
Las contribuciones generadas con IA son bienvenidas, siempre que una persona las haya revisado y aprobado.

# Pendiente

1. Monte Carlo Tree Search (`tic_tac_toe`)
2. ver jugar a los agentes: un modo `--watch` para los ejemplos, con una sola función de dibujo para dos vistas
   - en la terminal, con el clásico arte ASCII de siempre
   - en una ventana, con pixel art retro: 2D, dibujado a baja resolución y ampliado con píxeles nítidos,
     con sprites y una paleta de 16 colores definidos en el código (sin archivos de recursos), una fuente pixelada,
     sonidos chiptune opcionales generados en el código, y el aprendizaje mostrado con el mismo estilo (una curva de
     aprendizaje pixelada, los valores Q como colores de las casillas, y una línea de estado)
   - la ventana usaría [raylib](https://www.raylib.com) (licencia zlib, Linux y macOS), en un subproyecto de CMake
     opcional y desactivado por defecto, para que `relearn.hpp` y la compilación por defecto sigan sin dependencias

# Referencias

1. Sutton, R.S. y Barto, A.G., 2018. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), 2.ª edición. Cambridge, MA: MIT Press.
   Los números de sección, ejemplo y figura de arriba se refieren a esta edición, salvo los que dicen "1.ª ed.".
2. [Gymnasium](https://gymnasium.farama.org), el sucesor de OpenAI Gym, que sigue en mantenimiento activo.

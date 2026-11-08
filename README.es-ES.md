

![RELEARN](https://github.com/alexge233/relearn/blob/master/images/relearn.png?raw=true)

Una biblioteca de plantillas *solo encabezados* (header-only) de Aprendizaje por Refuerzo para C++14.
Minimalista y sencilla de usar, para una variedad de escenarios.
Se basa en [el libro de Sutton y Barto](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html) 
e implementa *parte* de la funcionalidad descrita en él.

Actualmente, la biblioteca implementa Q-Learning para sistemas deterministas, así como para sistemas no deterministas.

La regla de actualización de Q-Learning (tanto para políticas offline como online) es:

![Q(s_t,a_) = Q(s_t,a_t) + alpha * (R_t + gamma * max(Q_{s+t},a_t) - Q(s_t,a_t))](https://github.com/alexge233/relearn/blob/master/images/q_learning.png?raw=true)

La decisión de la política depende entirely de ti; en el caso de un sistema determinista,
usualmente las opciones son permanecer en la política:

![a_t = Π(s_t)](https://github.com/alexge233/relearn/blob/master/images/deterministic_policy.png?raw=true)

o, si se usa greedy explorativo online (ε-Greedy), asignar alguna *probabilidad decreciente* para usar la política
o tomar una acción aleatoria:

![P_{s_t}^{Π}(a_t)](https://github.com/alexge233/relearn/blob/master/images/stochastic_policy.png?raw=true)

Para Q-Learning en un sistema no determinista (p. ej., cuando la certeza de transitar de un estado a otro
no es constante), usamos una función de transición:

![f = f(s_t,s_{t+1})](https://github.com/alexge233/relearn/blob/master/images/transition_func.png?raw=true)

Esa función de transición sugiere que, desde el estado `s_t`, si se toma la acción `a_t`, el agente termina en el estado `s_{t+1}`.
La probabilidad asignada a esa transición debe ser **observada** después de que se realice:

![P(s_t,a_t)(f) = Σ P(s_t,a_t)(s_{t+1})](https://github.com/alexge233/relearn/blob/master/images/transition_prob.png?raw=true)

Esta probabilidad cuantifica la *verosimilitud* de la transición `s_t → s_{t+1}` para la acción `a_t`.
De manera similar, la *recompensa esperada* es:

![E(s_t,s_{t+1}) = Σ f(s_t,s_{t+1}) P(s_t,a_t)(f)](https://github.com/alexge233/relearn/blob/master/images/expected_reward.png?raw=true)

Esto simplemente establece que la recompensa es equivalente a la recompensa del estado resultante `s_{t+1}`, afectada por la probabilidad de transición.
Por lo tanto, la regla de actualización para sistemas probabilísticos es:

![Q(s_t,a_t) = E(r_{s_{t+1}}) + gamma * Σ maxQ(s_{t+1},a_t)P(s_t,a_t)(s_{t+1})](https://github.com/alexge233/relearn/blob/master/images/q_probabilistic.png?raw=true)

## Implementación en C++

La clase de plantilla `policy` permite el uso de tus propias clases `state` y `action`,
aunque se proporciona un par de clases envoltorio de plantilla `state` y `action` en el encabezado.
Puedes crear tus propias clases `state` y `action` y usarlas, pero deben cumplir con los siguientes criterios:
- los estados terminales deben tener una recompensa expuesta por un método `::reward()`
- los estados y acciones deben ser hashables (compatibles con hash)

Las clases `state` y `action` proporcionadas en el encabezado envuelven tus propias estructuras de estado y acción,
las cuales pueden ser cualquier cosa, siempre que sean hashables.
Ten cuidado de hacerlas únicas, de lo contrario el resultado puede ser un comportamiento indefinido.

### Proceso de Decisión de Markov

Un MDP es una secuencia de Estados  s<sub>t</sub> vinculados entre sí por acciones a<sub>t</sub>.
En el núcleo de `relearn` se encuentra la `policy` (política), que utiliza MDPs y los **mapea** usando la recompensa del estado final `R`.
Aunque puede haber múltiples episodios, y un episodio puede iterarse muchas veces, actualizarse, evaluarse y reexperimentarse, el proceso real se realiza en la clase `policy`. A continuación se muestra un resumen de un MDP.

![Markov Decision Process](https://github.com/alexge233/relearn/blob/master/images/mdp.png?raw=true)

Los parámetros de plantilla `state_trait` y `action_trait` se utilizan para describir qué es un estado y una acción,
los envoltorios se utilizan solo para indexación, hashing y cálculo de políticas.

El uso básico consiste en crear episodios, denominados `markov_chain`, que llenas según convenga,
y luego asignar una recompensa con un valor negativo (-1), neutro (0) o positivo (+1).

El uso de un `markov_chain` por defecto es un `std::deque<relearn::link<state_class,action_class>>`
donde `relearn::link` es la estructura que encapsula una observación de estado/acción.

Puedes reemplazar la clase contenedora o de enlace si lo deseas, pero la estructura `q_learning` requiere que:

- un episodio (un `markov_chain`) pueda ser iterado
- exista un método `::size()` para denotar el tamaño del episodio

Básicamente, la mayoría de los contenedores *STL* deberían funcionar directamente sin configuración adicional.
La biblioteca usa por defecto `double` para almacenar y actualizar los valores Q,
pero puedes modificarlo si es necesario.

La clase que más nos interesa es `policy`, que usa tus estados y acciones,
y los almacena en un mapa de estados, donde cada estado (clave) tiene como valor mapas (acción a valor):

```
.
├── s_t
|   ├── a_t: 0.03
|   └── a_t: -0.05
├── s_t
|   ├── a_t: -0.001
|   └── a_t: 0.9
```

Eres propietario y controlas los objetos de política, e incluso puedes usar múltiples, sin embargo ten en cuenta
que no están bloqueados para acceso MT (multihilo) (tienes que hacerlo manualmente).
Usando `q_learning` luego puedes establecer/actualizar los valores de la política iterando tus episodios.

# Dependencias

¡No hay **dependencias externas**!
Sin embargo, tu compilador **debe soportar C++14**, por lo que necesitarás:

- gcc 4.8.4 o superior
- clang 3.3 o superior

__nota__: ¡No lo he probado en una plataforma Windows!

__nota__: Si necesitas serialización, la rama `master` actual tiene una bandera `USING_BOOST_SERIALIZATION`,
la cual, cuando se establece en `ON`, habilitará la serialización de boost, siempre que tus estados y acciones (los parámetros de plantilla `state_trait` y `action_trait`) sean realmente serializables. 
En este caso, necesitas usar `find_package` de CMake para encontrar, incluir y enlazar correctamente `boost_serialization`.

# Compilación

¡No hay nada que compilar! Esta es una biblioteca solo encabezados, solo necesitas usar el encabezado **relearn.hpp**.
**Sin embargo**, puedes compilar los ejemplos para ver cómo se implementa la biblioteca.

Para hacerlo, simplemente:

```bash
mkdir build
cd build
cmake ..
make
```

### compilar ejemplos

Simplemente haz lo siguiente:

```bash
cmake .. -DBUILD_EXAMPLES=On
make
```

### compilar pruebas

```bash
cmake .. -DBUILD_TESTS=On
```

### habilitar serialización

Para habilitar la serialización, necesitas boost y boost-serialization.
Consulta tu distribución para saber cómo instalarlos.
Para habilitarlo, pasa una bandera a cmake:

```bash
cmake .. -DUSING_BOOST_SERIALIZATION=On
```

Por ejemplo, si quieres ejecutar las pruebas **con** serialización:

```bash
cmake .. -DUSING_BOOST_SERIALIZATION=On -DBUILD_EXAMPLES=On
```

También puedes establecer esta bandera en tu **propio** proyecto, si deseas guardar y cargar
políticas, estados o acciones.
Ten en cuenta que el `state_trait` (p. ej., tu **descriptor** de estado) y el 
`action_trait` (p. ej., tu **descriptor** de acción) deben **ser también serializables**.
Sobre cómo lograr esto, [echa un vistazo a este tutorial](http://www.boost.org/doc/libs/1_64_0/libs/serialization/doc/tutorial.html)
si no se cumple esta condición, terminarás con errores de compilación.

Debido a la flexibilidad de la serialización de boost, puedes guardar y cargar archivos binarios, de texto o xml.
Las versiones posteriores de boost soportan punteros inteligentes, por lo que incluso si tus descriptores son
`std::shared_ptr` o `std::unique_ptr` aún puedes guardarlos y cargarlos.

# Ejemplos

Hay una carpeta `examples` que estoy populando con ejemplos, comenzando con el típico problema de *gridworld*, 
y luego avanzando a un programa de *blackjack*.
Actualmente hay dos ejemplos de "Gridworld":
- un algoritmo on-policy offline: `examples/gridworld_offline.cpp` compilado como `ex_gridworld_offline`
- un algoritmo on-policy online: `examples/gridworld_online.cpp` compilado como `ex_gridworld_online`

## uso básico

La forma básica de usar la biblioteca es la siguiente:

1. crea una clase **state**, o usa una clase, estructura o PDT existente que describa (en un sentido *markoviano*) tu estado
2. crea una clase **action**, o usa una clase, estructura o PDT existente que describa la acción.
3. crea un *episodio* que por defecto es un `std::deque<relearn::link<state,action>>` que llenas y luego recompensas.

En este punto, dependiendo de si estás usando un algoritmo/enfoque **online** u **offline**, tienes las siguientes opciones:

4. sigue creando episodios, obtén una recompensa para el último/estado terminal y, una vez que hayas terminado, entrena la política con todos ellos,
**o**
5. cada vez que crees un episodio, obtén la recompensa y luego puedes entrenar tu política con ella.

Esa elección depende de ti y casi siempre depende del dominio, sistema o problema que estés intentando resolver.
Es por esta razón que no hay implementación de `on_policy` u `off_policy` o `e_greedy`, 
aquellos son algoritmos muy simples y específicos de la aplicación.

Echa un vistazo a los ejemplos de `gridworld`, que demuestran dos formas diferentes de lograr la misma tarea.
El ejemplo de `blackjack` es diferente: debido a que jugar Blackjack es una tarea de *probabilidad*, no podemos usar
un enfoque *determinista*, sino que usamos un enfoque *probabilístico*, en cuyo caso

1. tenemos que tomar un enfoque offline (no conocemos la transición de un estado a otro hasta que lo hemos experimentado)
2. tenemos que entrenar en probabilidades de transición (p. ej., no determinista)

## Gridworld

![gridworld image](https://github.com/alexge233/relearn/blob/master/images/gridworld.png?raw=true)

El cénit de la simplicidad cuando se trata de problemas de juguete de mundo de bloques/rejillas, nuestro agente reside en un gridworld de 10x10,
que está rodeado de bloques en los que no puede moverse (color negro).
El agente comienza en azul (x:1,y:8) y el objetivo es el verde (x:1,y:1).
Los bloques rojos son fuego/peligro/recompensa negativa, y hay un laberinto rudimentario.

Hay dos versiones del Gridworld, el enfoque offline:

- primero el agente explora aleatoriamente hasta que puede encontrar el bloque de rejilla de recompensa positiva (+1.0)
- luego actualiza sus políticas
- finalmente sigue la mejor política aprendida

Y el enfoque online:

- el agente explora aleatoriamente un episodio
- luego actualiza sus políticas
- luego lo intenta de nuevo, esta vez siguiendo políticas conocidas
- solo recurre a lo aleatorio cuando no existe una mejor acción *positiva*
- todo el proceso se repite hasta que se descubra el objetivo.

El gridworld real se guarda en un archivo de texto `gridworld.txt` (siéntete libre de cambiarlo).
El ejemplo `examples/gridworld_header.hpp` proporciona el código mínimo para demostrar este enfoque por etapas.
Los dos archivos:

- `examples/gridworld_offline.cpp`
- `examples/gridworld_online.cpp`

tienen las diferentes versiones de cómo puede resolverse esta tarea.

Una vez que hemos cargado el mundo (usando la función `populate`) establecemos el inicio en x:1, y:8 y luego
comenzamos la exploración.

### q-learning offline

La exploración offline se ejecuta infinitamente hasta que se encuentre el bloque de rejilla con una recompensa **positiva**.
Hasta que eso ocurra, el agente toma un enfoque *estocástico* (p. ej., aleatorio) y busca en el gridworld.
La función (el parámetro de plantilla `S` es estado y `A` es acción):

```cpp
template <typename S, 
          typename A>
std::deque<relearn::link<S,A>> explore(const world & w,
                                       std::mt19937 & gen,
                                       grid start);
```

hace lo siguiente:

- crea un nuevo episodio (p. ej., `relearn::markov_chain`)
- establece como estado raíz el bloque de rejilla inicial
- elige aleatoriamente una dirección (ver la estructura `rand_direction` para más detalles)
- repite esto hasta que (a) se encuentre una recompensa negativa (p. ej., pisó un bloque de fuego) o (b) se descubra el bloque objetivo

Cada vez que se llama al método `explore` en el bucle for de exploración principal,
se adquiere un episodio, que luego se anexa a la lista de episodios experimentados `std::vector<std::deque<relearn::link<state,action>>> episodes`.

Una vez que el agente ha encontrado el bloque de rejilla objetivo, actualiza su memoria de política usando esos episodios:

```cpp
auto learner = relearn::q_learning<state,action>{0.9, 0.1};
for (int k = 0; k < 10; k++) {
    for (auto episode : episodes) {
        learner(episode, policies);
    }
}
```

En esta instancia, el agente ejecutará diez iteraciones para cada episodio, y los parámetros reales de Q-Learning son:
- alpha (tasa de aprendizaje) 0.9
- gamma (descuento) 0.1

Finalmente, una vez que ha terminado la actualización de políticas, el agente procede a seguir (una vez) la mejor política de acción:

```cpp
on_policy(w, policies, start);
```

El resultado es que el agente irá directamente de x:1, y:8 a x:1, y:1 evitando cualquier bloque de fuego
y sin buscar innecesariamente en el gridworld.

Este es un escenario __determinista__, porque el agente sabe en cualquier momento, qué acción está tomando,
y a __qué__ estado lo llevará esa acción.

### q-learning online

La exploración online es algo diferente, porque el método `explore` hace lo siguiente:

- crea un nuevo episodio
- establece el estado raíz
- si existe una buena política (valor Q mayor que cero y un puntero de acción válido), los sigue
- de lo contrario, si no existe una buena política (o puntero de acción), toma una acción aleatoria
- repite hasta que se encuentre una recompensa, luego **entrena** las políticas con el último episodio

La diferencia aquí es que la exploración real se ve afectada instantáneamente por lo que ya se ha aprendido.
En comparación, el método offline no se ve afectado y puede repetir las mismas secuencias una y otra vez.
Sin embargo, si la versión online se detiene demasiado pronto, no hay garantía de que el agente haya aprendido la
ruta ideal u óptima hacia el objetivo, de hecho podría ser solo una ruta *mediocre* o *tonta* que haya descubierto.
Esto, por supuesto, también es un problema con el offline, donde la solución puede no descubrirse nunca.

Existen otros algoritmos más complejos (p. ej., e-greedy) donde el agente puede seguir la política,
pero elegir aleatoriamente ignorarla, para intentar descubrir una mejor solución.

## Blackjack

Un intento simplificado, donde un jugador usa probabilidades clásicas, el crupier (casa) simplemente roba hasta 17,
y el agente adaptativo usa Q-learning no determinista para jugar lo mejor posible.

El `state` es muy simple: una `hand` (mano) que se describe por el valor (valor mínimo y valor máximo, dependiendo de las cartas que se tengan).
El agente ignora la mano del crupier ya que eso aumentaría el espacio de estados,
así como la etiqueta o símbolo de las cartas que se tienen (siéntete libre de cambiar esto, simplemente adapta la función "hash" de `hand`).

Este ejemplo tarda mucho en ejecutarse, ya que el agente mapea las probabilidades de transición,
utilizando las observaciones de jugar múltiples partidas.

El archivo de encabezado `examples/blackjack_header.hpp` contiene las estructuras y métodos simples necesarios para jugar al blackjack,
mientras que el archivo fuente `examples/blackjack.cpp` tiene la lógica de alto nivel detrás de esto.

# TODO

1. implementar el algoritmo continuo de R-Learning
2. agregar trazas de elegibilidad (decay)

[1]: Sutton, R.S. y Barto, A.G., 1998. Aprendizaje por refuerzo: Una introducción (Vol. 1, No. 1). Cambridge: MIT press

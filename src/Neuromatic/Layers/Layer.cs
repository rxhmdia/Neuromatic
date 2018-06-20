using System;
using TensorFlow;

namespace Neuromatic.Layers
{
    /// <summary>
    /// This base class is used for all available layer types.
    /// You should derive from this class if you want to implement a custom layer type.
    /// </summary>
    public abstract class Layer
    {
        /// <summary>
        /// Initializes a new instance of <see cref="Layer"/>
        /// </summary>
        public Layer()
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="Layer"/>
        /// </summary>
        /// <param name="name">Name of the layer</param>
        public Layer(string name)
        {
            Name = name;
        }

        /// <summary>
        /// Gets the name of the layer
        /// </summary>
        public string Name { get; internal set; }

        /// <summary>
        /// Gets the configuration for the layer
        /// </summary>
        /// <remarks>Derived classes should assign their configuration
        /// to this property in order for the trainer to be able to train the final model</remarks>
        public LayerConfiguration Configuration { get; protected set; }

        /// <summary>
        /// Compiles the layer into a set of tensorflow operators
        /// </summary>
        /// <param name="graph">Tensorflow graph to use for compiling the layer</param>
        public abstract void Compile(TFGraph graph);

        /// <summary>
        /// Gets the output shape for the layer
        /// </summary>
        public abstract long[] OutputShape { get; }

        /// <summary>
        /// Gets whether this layer was compiled before.
        /// Primarily used by the model compilation process to determine the state of the graph.
        /// </summary>
        public bool Compiled => Configuration != null;
    }
}

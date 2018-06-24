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
        /// <para>
        /// Builds the layer by converting the abstract definition of the layer into 
        /// a concrete set of instructions for Tensorflow and a layer configuration
        /// for use when training the model.
        /// </para>
        /// <para>This method should register any parameters and initializers with the compilation context.
        /// So that they can be used during the training phase. </para>
        /// <para>Additionally you are required to store the layer configuration in the 
        /// <see cref="Configuration"/> property. This information is required as metadata 
        /// when the model is used.</para>
        /// <param name="context">Use this context to register trainable parameters
        /// and build the computational graph for the layer</param>
        public abstract TFOutput Compile(ModelCompilationContext context);

        /// <summary>
        /// Gets the output shape for the layer
        /// </summary>
        public abstract long[] OutputShape { get; }

        /// <summary>
        /// Gets or sets the layer configuration for the layer
        /// </summary>
        public LayerConfiguration Configuration { get; protected set; }
    }
}

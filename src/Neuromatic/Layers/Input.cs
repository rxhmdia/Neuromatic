using System;
using System.Runtime.Serialization;
using TensorFlow;

namespace Neuromatic.Layers
{
    /// <summary>
    /// Defines a single input layer for a neural network.
    /// </summary>
    public class Input : Layer
    {
        private readonly long[] _shape;

        /// <summary>
        /// Initializes a new instance of <see cref="Input"/>
        /// </summary>
        /// <param name="shape">Shape of the layer</param>
        public Input(long[] shape)
        {
            _shape = shape;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="Input"/>
        /// </summary>
        /// <param name="name">Name of the layer</param>
        /// <param name="shape">Shape of the layer</param>
        public Input(long[] shape, string name) : base(name)
        {
            _shape = shape;
        }

        /// <summary>
        /// Gets the output shape for the layer
        /// </summary>
        public override long[] OutputShape
        {
            get
            {
                long[] outputShape = new long[_shape.Length + 1];

                // First dimension is always unknown so that we can feed batches of data
                outputShape[0] = -1;

                for (int index = 0; index < _shape.Length; index++)
                {
                    outputShape[index + 1] = _shape[index];
                }

                return outputShape;
            }
        }

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
        public override TFOutput Compile(ModelCompilationContext context)
        {
            if (Configuration != null)
            {
                return Configuration.Output;
            }

            if (_shape.Length == 0)
            {
                throw new ModelCompilationException("Shape must have at least one dimension");
            }

            var placeholder = context.Graph.Placeholder(TFDataType.Double, 
                new TFShape(OutputShape), operName: Name);

            Configuration = new LayerConfiguration(new TFOutput[] { }, new TFOperation[] {  }, placeholder);

            return placeholder;
        }
    }
}

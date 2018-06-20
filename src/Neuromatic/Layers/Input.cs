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
        /// <para>This method should assign the resulting layer configuration to the <see cref="Configuration"/>
        /// property of the layer. This configuration is used by the model to determine the trainable parameters,
        /// the output of the layer and any initializers that need to run as part of the training process.</para>
        /// </summary>
        /// <param name="graph">Graph to add the instructions to</param>
        public override void Compile(TFGraph graph)
        {
            if (_shape.Length == 0)
            {
                throw new ModelCompilationException("Shape must have at least one dimension");
            }

            var placeholder = graph.Placeholder(TFDataType.Double, new TFShape(OutputShape), operName: Name);
            Configuration = new LayerConfiguration(new TFOutput[] { }, placeholder, new TFOperation[] { });
        }
    }
}

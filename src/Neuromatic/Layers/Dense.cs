using System;
using System.Collections.Generic;
using System.Text;
using Neuromatic.Core;
using TensorFlow;

namespace Neuromatic.Layers
{
    /// <summary>
    /// A dense layer connects all its inputs to a defined number of outputs. 
    /// You can control the density of the connections with weights and a bias term.
    /// The output of each of the connections is passed through an activation function to
    /// control the strength of the output signal.
    /// </summary>
    public class Dense : Layer
    {
        private long[] _outputShape;

        /// <summary>
        /// Initializes a new instance of <see cref="Dense"/>
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <param name="input"></param>
        public Dense(int units, ActivationFunction activation, Layer input, string name) : base(name)
        {
            Units = units;
            Input = input;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="Dense"/>
        /// </summary>
        /// <param name="name"></param>
        /// <param name="units"></param>
        /// <param name="activation"></param>
        /// <param name="kernelInitializer"></param>
        /// <param name="input"></param>
        public Dense(int units,
            ActivationFunction activation,
            InitializationFunction kernelInitializer,
            InitializationFunction biasInitializer,
            Layer input, string name) : base(name)
        {
            Units = units;
            Input = input;
            KernelInitializer = kernelInitializer;
            BiasInitializer = biasInitializer;
            Activation = activation;
        }

        /// <summary>
        /// Gets the shape of the layer
        /// </summary>
        public override long[] Shape => _outputShape;

        /// <summary>
        /// Gets the number of units in the dense layer
        /// </summary>
        public int Units { get; }

        /// <summary>
        /// Gets the input for the layer
        /// </summary>
        public Layer Input { get; }

        /// <summary>
        /// Gets the initializer for the weights
        /// </summary>
        public InitializationFunction KernelInitializer { get; }

        /// <summary>
        /// Gets the initializer for the bias term
        /// </summary>
        public InitializationFunction BiasInitializer { get; }

        /// <summary>
        /// Gets the activation function
        /// </summary>
        public ActivationFunction Activation { get; }

        /// <summary>
        /// Compiles the dense layer
        /// </summary>
        /// <param name="backend">Backend to use for performing compilation</param>
        /// <returns>Returns the compiled layer output node</returns>
        public override ExecutableModelNode Compile(ModelBackend backend)
        {
            _outputShape = CalculateOutputShape();

            var kernelInitializer = KernelInitializer ?? backend.Initializers.RandomNormal();
            var biasInitializer = BiasInitializer ?? backend.Initializers.RandomNormal();
            var activation = Activation ?? backend.Activations.Sigmoid();

            var weights = backend.Weights(new long[] { Input.Shape[1], Units }, kernelInitializer, $"{Name}_Weights");
            var bias = backend.Weights(new long[] { Units, -1 }, biasInitializer, $"{Name}_Bias");

            var output = backend.Dot(Input.Compile(backend), weights, $"{Name}_MatMul");
            output = backend.BiasAdd(output, bias);

            return activation.Create(output);
        }

        /// <summary>
        /// Calculates the output shape for the layer
        /// </summary>
        /// <returns>Returns the shape of the layer</returns>
        long[] CalculateOutputShape()
        {
            long[] outputShape = new long[Input.Shape.Length];

            for (int index = 0; index < outputShape.Length - 1; index++)
            {
                outputShape[index] = Input.Shape[index];
            }

            outputShape[outputShape.Length - 1] = Units;

            return outputShape;
        }
    }
}

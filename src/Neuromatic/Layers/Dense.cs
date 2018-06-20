using Neuromatic.Initializers;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace Neuromatic.Layers
{
    /// <summary>
    /// <para>
    /// Dense layers are used to create fully connected layers in neural networks.
    /// The layer implements the function output = activation(dot(input,kernel) + bias).
    /// </para>
    /// <para>
    /// <b>activation</b> is the element-wise activation function applied to the summed inputs. 
    /// <b>kernel</b> is a set of weights initialized using the weights initialization function.
    /// <b>bias</b> is a vector initialized using the bias initialization function.
    /// </para>
    /// </summary>
    public class Dense : Layer
    {
        private int _units;
        private InitializationFunction _weightsInitializer;
        private InitializationFunction _biasInitializer;
        private Layer _input;

        /// <summary>
        /// Initializes a new instance of <see cref="Dense"/>
        /// </summary>
        /// <param name="units">Kernel size of the layer</param>
        /// <param name="input">Input layer to connect to</param>
        /// <param name="biasInitializer">Initialization function for the bias vector</param>
        /// <param name="weightsInitializer">Initialization function for the weights matrix</param>
        public Dense(int units, InitializationFunction weightsInitializer, InitializationFunction biasInitializer, Layer input)
        {
            if (_units <= 0)
            {
                throw new ArgumentException("Invalid layer size. Should be greater than zero", nameof(units));
            }

            if (_weightsInitializer == null)
            {
                throw new ArgumentNullException(nameof(weightsInitializer));
            }

            if (_biasInitializer == null)
            {
                throw new ArgumentNullException(nameof(biasInitializer));
            }

            if (_input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            _units = units;
            _weightsInitializer = weightsInitializer;
            _biasInitializer = biasInitializer;
            _input = input;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="Dense"/>
        /// </summary>
        /// <param name="units">Kernel size of the layer</param>
        /// <param name="input">Input layer to connect to</param>
        /// <param name="biasInitializer">Initialization function for the bias vector</param>
        /// <param name="weightsInitializer">Initialization function for the weights matrix</param>
        /// <param name="name">Name of the layer</param>
        public Dense(int units, InitializationFunction weightsInitializer, InitializationFunction biasInitializer, Layer input, string name) : base(name)
        {
            if (units <= 0)
            {
                throw new ArgumentException("Invalid layer size. Should be greater than zero", nameof(units));
            }

            _units = units;
            _weightsInitializer = weightsInitializer ?? throw new ArgumentNullException(nameof(weightsInitializer));
            _biasInitializer = biasInitializer ?? throw new ArgumentNullException(nameof(biasInitializer));
            _input = input ?? throw new ArgumentNullException(nameof(input));
        }

        /// <summary>
        /// Initializes a new instance of <see cref="Dense"/>
        /// </summary>
        /// <param name="units">Kernel size of the layer</param>
        /// <param name="input">Input layer to connect to</param>
        /// <param name="name">Name of the layer</param>
        public Dense(int units, Layer input, string name) : base(name)
        {
            if (units <= 0)
            {
                throw new ArgumentException("Invalid layer size. Should be greater than zero", nameof(units));
            }

            _units = units;
            _input = input ?? throw new ArgumentNullException(nameof(input));
            _weightsInitializer = new RandomNormal();
            _biasInitializer = new RandomNormal();
        }

        /// <summary>
        /// Initializes a new instance of <see cref="Dense"/>
        /// </summary>
        /// <param name="units">Kernel size of the layer</param>
        /// <param name="input">Input layer to connect to</param>
        public Dense(int units, Layer input)
        {
            _units = units;
            _input = input;
            _weightsInitializer = new RandomNormal();
            _biasInitializer = new RandomNormal();
        }

        /// <summary>
        /// Gets the output shape for the layer
        /// </summary>
        public override long[] OutputShape
        {
            get
            {
                long[] shape = new long[_input.OutputShape.Length];

                for (int index = 0; index < _input.OutputShape.Length - 1; index++)
                {
                    shape[index] = _input.OutputShape[index];
                }

                shape[shape.Length - 1] = _units;

                return shape;
            }
        }

        /// <summary>
        /// Compiles the dense layer
        /// </summary>
        /// <param name="graph">Graph to use for compiling the dense layer</param>
        public override void Compile(TFGraph graph)
        {
            if (!_input.Compiled)
            {
                _input.Compile(graph);
            }

            var inputDimension = _input.OutputShape[_input.OutputShape.Length - 1];

            using (var scope = graph.WithScope(Name))
            {
                TFShape weightsShape = new TFShape(inputDimension, _units);
                TFShape biasShape = new TFShape(_units);

                var weights = graph.VariableV2(
                    weightsShape,
                    TFDataType.Double, operName: "Weights");

                var bias = graph.VariableV2(
                    biasShape,
                    TFDataType.Double, operName: "Bias");

                var initializers = new[]
                {
                    graph.Assign(weights, _weightsInitializer.Compile(graph, weightsShape)).Operation,
                    graph.Assign(bias, _biasInitializer.Compile(graph, biasShape)).Operation
                };

                // Formula: input * W + b
                // TODO: Make sure that we can work without a bias term.
                var output = graph.Add(graph.MatMul(_input.Configuration.Output, weights), bias);

                Configuration = new LayerConfiguration(new[] { weights, bias }, output, initializers);
            }
        }
    }
}

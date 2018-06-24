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
            var input = _input.Compile(context);

            var inputDimension = _input.OutputShape[_input.OutputShape.Length - 1];

            using (var scope = context.Graph.WithScope(Name))
            {
                TFShape weightsShape = new TFShape(inputDimension, _units);
                TFShape biasShape = new TFShape(_units);

                var weights = context.Graph.VariableV2(
                    weightsShape,
                    TFDataType.Double, operName: "Weights");

                var bias = context.Graph.VariableV2(
                    biasShape,
                    TFDataType.Double, operName: "Bias");

                var initializers = new[]
                {
                    context.Graph.Assign(weights, _weightsInitializer.Compile(context.Graph, weightsShape)).Operation,
                    context.Graph.Assign(bias, _biasInitializer.Compile(context.Graph, biasShape)).Operation
                };

                // Formula: input * W + b
                // TODO: Make sure that we can work without a bias term.
                var output = context.Graph.Add(context.Graph.MatMul(input, weights), bias);

                context.AddParameters(weights, bias);
                context.AddInitializers(initializers);

                Configuration = new LayerConfiguration(new[] {  weights,bias}, initializers, output);

                return output;
            }
        }
    }
}

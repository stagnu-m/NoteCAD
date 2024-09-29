using System;
using Google.OrTools.Init;
using Google.OrTools.LinearSolver;
using LinearSolver = Google.OrTools.LinearSolver;


namespace Assets.Code.Solver
{
    public class LinearSolverExample
    {
        private const bool _debug = true;

        public static void SolveLinearProgram(double[,] A, double[] B, ref double[] X)
        {
	        UnityEngine.Profiling.Profiler.BeginSample("LinearProgram.Solve");
            Console.WriteLine("Google.OrTools version: " + Google.OrTools.Init.OrToolsVersion.VersionString());

            // Create the linear solver with the GLOP backend.
            LinearSolver.Solver solver = LinearSolver.Solver.CreateSolver("GLOP");

            if (solver is null)
            {
                Console.WriteLine("Could not create solver GLOP");
                return;
            }

            int numVars = X.Length;
            int numConstraints = B.Length;

            if (_debug)
            {
                Console.WriteLine($"Number of variables: {numVars}");
                Console.WriteLine($"Number of constraints: {numConstraints}");
            }

            // Create variables u and v
            Variable[] uVars = new Variable[numVars];
            Variable[] vVars = new Variable[numVars];
            double infinity = double.PositiveInfinity;

            for (int i = 0; i < numVars; i++)
            {
                uVars[i] = solver.MakeNumVar(0.0, infinity, $"u{i}");
                vVars[i] = solver.MakeNumVar(0.0, infinity, $"v{i}");
                solver.Objective().SetCoefficient(uVars[i], 1.0);
                solver.Objective().SetCoefficient(vVars[i], 1.0);
                //uVars[i].SetObjectiveCoefficient(1.0);
                //vVars[i].SetObjectiveCoefficient(1.0);
            }

            solver.Objective().SetMinimization();

            // Add constraints
            for (int j = 0; j < numConstraints; j++)
            {
               LinearSolver.Constraint constraint = solver.MakeConstraint(B[j], B[j], $"c{j}");

                for (int k = 0; k < numVars; k++)
                {
                    double coefficient = A[j, k];
                    if (coefficient != 0.0)
                    {
                        constraint.SetCoefficient(uVars[k], coefficient);
                    }
                }

                for (int k = 0; k < numVars; k++)
                {
                    double coefficient = -A[j, k];
                    if (coefficient != 0.0)
                    {
                        constraint.SetCoefficient(vVars[k], coefficient);
                    }
                }
            }

            LinearSolver.Solver.ResultStatus resultStatus = solver.Solve();

            if (resultStatus == LinearSolver.Solver.ResultStatus.OPTIMAL || resultStatus == LinearSolver.Solver.ResultStatus.FEASIBLE)
            {
                for (int i = 0; i < numVars; i++)
                {
                    double u = uVars[i].SolutionValue();
                    double v = vVars[i].SolutionValue();
                    if (_debug)
                    {
                        Console.WriteLine($"{uVars[i].Name()} = {u}");
                        Console.WriteLine($"{vVars[i].Name()} = {v}");
                    }
                    X[i] = u - v;
                }

                if (_debug)
                {
                    Console.WriteLine("Solution:");
                    for (int i = 0; i < numVars; i++)
                    {
                        Console.WriteLine($"X[{i}] = {X[i]}");
                    }
                }
            }
            else
            {
                Console.WriteLine("No solution found.");
            }

            UnityEngine.Profiling.Profiler.EndSample();

        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using Assets.Code.Tools;
using Google.OrTools.LinearSolver;
using UnityEngine;
using LinearSolver = Google.OrTools.LinearSolver;

namespace Assets.Code.Solver
{
    public class LinearSolverExample
    {
        private const bool _debug = true;
        public static bool IsOldNorm => ChooseNormComponent.IsOldNorm();
        private static IndexToBlock indexToBlock { get; set; } = new();
        private static List<IndexToBlock> indicesToConsider { get; set; } = new();
        public static void SolveLinearProgram(double[,] A, double[] B, ref double[] X,
                                              List<int> dragIndices, int dof)
        {
            UnityEngine.Profiling.Profiler.BeginSample("LinearProgram.Solve");
            var copy_A = new double[A.GetLength(0), A.GetLength(1)];
            var copy_B = new double[B.Length];
            var copy_X = new double[X.Length];
            var lastOptimalSolution = new double[X.Length];
            indicesToConsider = new List<IndexToBlock>();
            do
            {
                indexToBlock = indicesToConsider.FirstOrDefault(x => !x.IsUsed);

                copy_A = DeepCopyMatrix(A);
                copy_B = DeepCopyArray(B);
                copy_X = DeepCopyArray(X);
                //Debug.Log("Google.OrTools version: " + Google.OrTools.Init.OrToolsVersion.VersionString());

                // Create the linear solver with the GLOP backend.
                LinearSolver.Solver solver = LinearSolver.Solver.CreateSolver("GLOP");

                if (solver is null)
                {
                    Debug.Log("Could not create solver GLOP");
                    return;
                }

                int numVars = copy_X.Length;
                int numConstraints = copy_B.Length;

                if (_debug)
                {
                    //Debug.Log($"Number of variables: {numVars} \n");
                    //Debug.Log($"Number of constraints: {numConstraints} \n");
                    Debug.Log($"Drag points:{string.Join(", ", dragIndices)} \n");
                }

                // Create variables u and v for the positive and negative components of X
                Variable[] uVars = new Variable[numVars];
                Variable[] vVars = new Variable[numVars];
                double infinity = double.PositiveInfinity;

                Objective objective = solver.Objective();

                for (int i = 0; i < numVars; i++)
                {
                    bool isBlock = false;
                    if (indexToBlock != null)
                    {
                        indexToBlock.IsUsed = true;
                        isBlock = indexToBlock.Index_x == i || indexToBlock.Index_y == i;
                    }

                    var infinityOrZero = isBlock ? 0.0 : infinity;
                    uVars[i] = solver.MakeNumVar(0.0, infinityOrZero, $"u{i}");
                    vVars[i] = solver.MakeNumVar(0.0, infinityOrZero, $"v{i}");
                    if (IsOldNorm)
                    {
                        objective.SetCoefficient(uVars[i], 1.0);
                        objective.SetCoefficient(vVars[i], 1.0);
                    }
                }

                if (!IsOldNorm)
                {
                    ImprovedL1(numVars, solver, infinity, objective, uVars, vVars);
                }

                objective.SetMinimization();

                // Add the original constraints from A and B
                for (int j = 0; j < numConstraints; j++)
                {
                    LinearSolver.Constraint constraint = solver.MakeConstraint(copy_B[j], copy_B[j], $"c{j}");

                    for (int k = 0; k < numVars; k++)
                    {
                        double coefficient = copy_A[j, k];
                        if (coefficient != 0.0)
                        {
                            constraint.SetCoefficient(uVars[k], coefficient);
                        }
                    }

                    for (int k = 0; k < numVars; k++)
                    {
                        double coefficient = -copy_A[j, k];
                        if (coefficient != 0.0)
                        {
                            constraint.SetCoefficient(vVars[k], coefficient);
                        }
                    }
                }

                LinearSolver.Solver.ResultStatus resultStatus = solver.Solve();

                if (resultStatus == LinearSolver.Solver.ResultStatus.OPTIMAL ||
                    resultStatus == LinearSolver.Solver.ResultStatus.FEASIBLE)
                {
                    for (int i = 0; i < numVars; i++)
                    {
                        double u = uVars[i].SolutionValue();
                        double v = vVars[i].SolutionValue();
                        copy_X[i] = u - v;
                    }

                    lastOptimalSolution = DeepCopyArray(copy_X);

                }
                else
                {
                    Debug.Log("No solution found.");
                }

                BlockFreeCoordinates(copy_X, dragIndices, dof, numVars);


            }
            while (indicesToConsider.Count(x => !x.IsUsed) > 0);

            A = copy_A;
            B = copy_B;
            X = lastOptimalSolution;
            UnityEngine.Profiling.Profiler.EndSample();

        }

        public static void SolveLinearProgramInitial(
	        ref double[,] A, 
	        ref double[] B, 
	        ref double[] X,
	        List<int> dragIndices, 
	        int dof, 
	        IndexToBlock indexToBlock = null)
        {
            UnityEngine.Profiling.Profiler.BeginSample("LinearProgram.Solve");
            //Debug.Log("Google.OrTools version: " + Google.OrTools.Init.OrToolsVersion.VersionString());

            // Create the linear solver with the GLOP backend.
            LinearSolver.Solver solver = LinearSolver.Solver.CreateSolver("GLOP");

            if (solver is null)
            {
                Debug.Log("Could not create solver GLOP");
                return;
            }

            int numVars = X.Length;
            int numConstraints = B.Length;

            if (_debug)
            {
                //Debug.Log($"Number of variables: {numVars} \n");
                //Debug.Log($"Number of constraints: {numConstraints} \n");
                Debug.Log($"Drag points:{string.Join(", ", dragIndices)} \n");
            }

            // Create variables u and v for the positive and negative components of X
            Variable[] uVars = new Variable[numVars];
            Variable[] vVars = new Variable[numVars];
            double infinity = double.PositiveInfinity;

            Objective objective = solver.Objective();

            for (int i = 0; i < numVars; i++)
            {
	            bool isBlock = false;
	            if (indexToBlock != null)
	            {
		            indexToBlock.IsUsed = true;
		            isBlock = indexToBlock.Index_x == i || indexToBlock.Index_y == i;
	            }

	            var infinityOrZero = isBlock ? 0.0 : infinity;
                uVars[i] = solver.MakeNumVar(0.0, infinityOrZero, $"u{i}");
                vVars[i] = solver.MakeNumVar(0.0, infinityOrZero, $"v{i}");
                if (IsOldNorm)
                {
                    objective.SetCoefficient(uVars[i], 1.0);
                    objective.SetCoefficient(vVars[i], 1.0);
                }
            }

            if (!IsOldNorm)
            {
                ImprovedL1(numVars, solver, infinity, objective, uVars, vVars);
            }

            objective.SetMinimization();

            // Add the original constraints from A and B
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

            if (resultStatus == LinearSolver.Solver.ResultStatus.OPTIMAL ||
                resultStatus == LinearSolver.Solver.ResultStatus.FEASIBLE)
            {
                for (int i = 0; i < numVars; i++)
                {
                    double u = uVars[i].SolutionValue();
                    double v = vVars[i].SolutionValue();
                    X[i] = u - v;
                }
            }
            else
            {
                Debug.Log("No solution found.");
            }
            UnityEngine.Profiling.Profiler.EndSample();

        }

        #region Private Methods

        //-----------------------------------------------------------------------------------------------------------------------------

        #region Improving functions

        private static void BlockFreeCoordinates(double[] X, List<int> dragIndices, int dof, int numVars)
        {
            // Number of points in X (each point has two components)
            int numPairs = numVars / 2;

            // If no DOF, exit early
            if (dof == 0) return;

            if (indicesToConsider.Count >= dof) return;

            // Iterate over all pairs (points) in X
            for (int i = 0; i < numPairs; i++)
            {
                // Skip drag indices
                int varIndex_x = 2 * i;
                int varIndex_y = 2 * i + 1;
                if (dragIndices.Contains(varIndex_x) || dragIndices.Contains(varIndex_y)) continue;

                // Current point components
                double x_coordinate = X[varIndex_x];
                double y_coordinate = X[varIndex_y];

                // Skip if both coordinates are zero or both are non-zero
                if (x_coordinate == 0 ^ y_coordinate == 0)
                {
                    //TODO
                    indicesToConsider.Add(new IndexToBlock
                    {
                        Index_x = varIndex_x,
                        Value_x = x_coordinate,
                        Index_y = varIndex_y,
                        Value_y = y_coordinate,
                        IsUsed = false
                    });
                }
                indicesToConsider.Sort((a, b) => a.AbsoluteSum.CompareTo(b.AbsoluteSum));
            }

            //indicesToConsider = indicesToConsider.OrderBy(x => x.AbsoluteSum).ToList();

        }

        private static void ImprovedL1(
            int numVars,
            LinearSolver.Solver solver,
            double infinity,
            Objective objective,
            Variable[] uVars,
            Variable[] vVars,
            IndexToBlock indexToBlock = null
        )
        {
            // Number of pairs in X (assuming numVars is even)
            int numPairs = numVars / 2;
            Variable[] zVars = new Variable[numPairs];

            // Create auxiliary variables z for max(|X[2*i]|, |X[2*i+1]|)
            for (int i = 0; i < numPairs; i++)
            {
	            bool isBlock = false;
	            if (indexToBlock != null)
	            {
		            indexToBlock.IsUsed = true;
		            isBlock = indexToBlock.Index_x == 2*i && indexToBlock.Index_y == 2*i+1;
	            }
	            var infinityOrZero = isBlock ? 0.0 : infinity;
                zVars[i] = solver.MakeNumVar(0.0, infinityOrZero, $"z{i}");
                objective.SetCoefficient(zVars[i], 1.0);
            }

            // Add constraints to ensure z[i] >= |X[2*i]| and z[i] >= |X[2*i + 1]|
            for (int i = 0; i < numPairs; i++)
            {
                int varIndex1 = 2 * i;
                int varIndex2 = 2 * i + 1;

                // Constraints to ensure zVars[i] >= absolute value of X[2*i] (|X[2*i]|)
                solver.Add(zVars[i] >= uVars[varIndex1] + vVars[varIndex1]);

                // Constraints to ensure zVars[i] >= absolute value of X[2*i + 1] (|X[2*i + 1]|)
                solver.Add(zVars[i] >= uVars[varIndex2] + vVars[varIndex2]);
            }
        }

        #endregion

        //-----------------------------------------------------------------------------------------------------------------------------

        #region Helper functions

        private static double[,] DeepCopyMatrix(double[,] original)
        {
            int rows = original.GetLength(0);
            int cols = original.GetLength(1);
            double[,] copy = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    copy[i, j] = original[i, j];
                }
            }
            return copy;
        }

        private static double[] DeepCopyArray(double[] original)
        {
            double[] copy = new double[original.Length];

            for (int i = 0; i < original.Length; i++)
            {
                copy[i] = original[i];
            }
            return copy;
        }

        #endregion

        //-----------------------------------------------------------------------------------------------------------------------------

        #endregion

    }

    public class IndexToBlock
    {
        public int Index_x { get; set; }
        public double Value_x { get; set; }
        public int Index_y { get; set; }
        public double Value_y { get; set; }
        public bool IsUsed { get; set; }
        public double AbsoluteSum => Math.Abs(Value_x) + Math.Abs(Value_y);
    }


}

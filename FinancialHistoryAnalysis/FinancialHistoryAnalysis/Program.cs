using Google.OrTools.LinearSolver;
using Google.OrTools.Sat;

namespace FinancialHistoryAnalysis
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //定义单年最大允许亏损比例。（比如：0.2代表单年最大允许亏损比例为20%；1代表无限制；0代表不允许亏损）
            float allowableMaximumLossRatio = 1f; //无限制
            //float allowableMaximumLossRatio = 0.2f; //单年最大允许亏损比例为20%
            //float allowableMaximumLossRatio = 0f; //不允许亏损

            //待处理数据，此处全部转换为整数处理
            (String year, long[] values)[] data = new[]
            {
                ("2005", new long[]{ 10273, 10236, 10912, 10140, 8848}),
                ("2006", new long[]{ 10280, 10150, 11494, 22263, 21190}),
                ("2007", new long[]{ 10360, 10336, 11822, 22833, 26621}),
                ("2008", new long[]{ 11542, 10356, 10646, 4858, 3708}),
                ("2009", new long[]{ 10425, 10142, 10504, 17117, 20547}),
                ("2010", new long[]{ 10392, 10181, 10690, 9972, 9312}),
                ("2011", new long[]{ 10463, 10355, 9711, 7547, 7759}),
                ("2012", new long[]{ 10588, 10397, 10622, 10545, 10468}),
                ("2013", new long[]{ 10482,10395,10061,11013,10544}),
                ("2014", new long[]{ 10597,10460,11848,12939,15244}),
                ("2015", new long[]{ 10556,10362,10993,13467,13850}),
                ("2016", new long[]{ 10471,10261,9965,8969,8709}),
                ("2017", new long[]{ 10422,10384,10165,11063,10493}),
                ("2018", new long[]{ 10496,10375,10543,7683,7175}),
                ("2019", new long[]{ 10446,10266,10422,14109,13302}),
                ("2020", new long[]{ 10414,10213,10315,14454,12343}),
                ("2021", new long[]{ 10310, 10228, 10393, 10587, 10917}),
                ("2022", new long[]{ 10350, 10101, 10090, 8928, 9047}),
            };

            // 创建CP模型.
            CpModel model = new CpModel();

            //定义变量：各类资产配置比例
            IntVar a = model.NewIntVar(0, 100, "a"); //银行理财
            IntVar b = model.NewIntVar(0, 100, "b"); //货币基金
            IntVar c = model.NewIntVar(0, 100, "c"); //债卷基金
            IntVar d = model.NewIntVar(0, 100, "d"); //股票基金
            IntVar e = model.NewIntVar(0, 100, "e"); //股票

            //创建约束条件：配置比例总和为100%
            model.Add(a + b + c + d + e <= 100);
            model.Add(a + b + c + d + e >= 100);

            //创建约束条件：限定低风险配置比例
            //model.Add(a >= 40);
            //model.Add(d + e <= 40);

            //定义变量数组：单年年末资金
            IntVar[] yearResults = new IntVar[data.Length];
            //定义变量数组：单年收益率
            IntVar[] yearRatios = new IntVar[data.Length];

            for (int i = 0; i<data.Length; i++)
            {
                var yearItem = data[i];

                //定义变量：当前年度收益率
                IntVar ratio = model.NewIntVar(0, 100 * 10000 * 3, $"ratio{i}");
                model.Add(ratio == a * yearItem.values[0] + b * yearItem.values[1] + c * yearItem.values[2] + d * yearItem.values[3] + e * yearItem.values[4]);
                yearRatios[i] = ratio;

                //创建约束条件：单年最大允许亏损比例
                model.Add(ratio >= Convert.ToInt32(100 * (1 - allowableMaximumLossRatio)) * 10000);

                //定义变量：当前年末资金
                IntVar resultA = model.NewIntVar(0, 100 * 100 * 10000 * Convert.ToInt64(Math.Pow(3, i+1)), $"resultA{i}");
                model.AddMultiplicationEquality(resultA, i==0? model.NewConstant(100) : yearResults[i-1], ratio);

                //定义变量：由于原生数据的收益率和配置比例是使用转换后的整数计算的，所以这里使用当前年末资金除以100*10000
                IntVar result = model.NewIntVar(0, 100 * Convert.ToInt64(Math.Pow(3, i+1)), $"result{i}");
                model.AddDivisionEquality(result, resultA, model.NewConstant(100 * 10000));
                yearResults[i] = result;
            }

            //设定求解目标为最终资金最大
            model.Maximize(yearResults[data.Length -1]);

            //求解
            CpSolver solver = new CpSolver();
            CpSolverStatus status = solver.Solve(model);

            //输出求解结果
            if (status == CpSolverStatus.Optimal || status == CpSolverStatus.Feasible)
            {
                Console.WriteLine("银行理财配置比: " + solver.Value(a)+"%");
                Console.WriteLine("货币基金配置比: " + solver.Value(b)+"%");
                Console.WriteLine("债卷基金配置比: " + solver.Value(c)+"%");
                Console.WriteLine("股票基金配置比: " + solver.Value(d)+"%");
                Console.WriteLine("股票配置比: " + solver.Value(e)+"%");

                for (int i = 0; i<data.Length; i++)
                {
                    Console.WriteLine($"{data[i].year} 年末资金:{solver.Value(yearResults[i])} 收益率:{String.Format("{0:P}", solver.Value(yearRatios[i]) / 1000000.00 - 1)}");
                }

                Console.WriteLine($"最终资金: {solver.ObjectiveValue}");
                Console.WriteLine($"年化收益率: {String.Format("{0:P}", Math.Pow((solver.ObjectiveValue - 100)/100, 1.00/data.Length)-1)}");
            }
            else
            {
                Console.WriteLine("求解失败，未找到合适结果.");
            }

            Console.WriteLine($"求解耗时: {solver.WallTime()}s");
        }
    }
}
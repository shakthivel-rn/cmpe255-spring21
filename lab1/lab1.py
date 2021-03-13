import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep='\t')
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        print('\n')
        print('Top x number of entries from the dataset and display as markdown format')
        print('---------------------------------------------------------------------------\n')
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        print('\n')
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        print('The number of observations/entries in the dataset')
        print('---------------------------------------------------------------------------\n')
        return self.chipo['order_id'].count()
    
    def info(self) -> None:
        # TODO
        # print data info.
        print('Print data info')
        print('---------------------------------------------------------------------------\n')
        print(self.chipo.info(verbose=True))
        print('\n')
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        print('Return the number of columns in the dataset')
        print('---------------------------------------------------------------------------\n')
        return len(self.chipo.columns)
        print('\n')
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        print('Print the name of all the columns')
        print('---------------------------------------------------------------------------\n')
        print(self.chipo.columns.values.tolist())
        print('\n')
    
    def most_ordered_item(self):
        # TODO
        print('Most ordered item')
        print('---------------------------------------------------------------------------\n')
        order_count = self.chipo.groupby(['item_name']).sum()
        order_count = order_count[order_count.quantity == order_count.quantity.max()]
        item_name = order_count.index.values[0] 
        quantity = int(order_count.reset_index()['quantity'])
        print(item_name)
        print(quantity)
        print('\n')
        return item_name, quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
        print('How many items were orderd in total?')
        print('---------------------------------------------------------------------------\n')
        print(self.chipo['quantity'].sum())
        print('\n')
        return self.chipo['quantity'].sum()
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        print('Totals Sales')
        print('---------------------------------------------------------------------------\n')
        self.chipo['item_price'] = self.chipo['item_price'].apply(lambda x: float(x[1:]))
        each_sale = self.chipo['item_price'] * self.chipo['quantity']
        self.total_value = each_sale.sum()
        print(self.total_value)
        print('\n')
        return self.total_value
   
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        print('How many orders were made in the dataset?')
        print('---------------------------------------------------------------------------\n')
        self.total_order = self.chipo['order_id'].max()
        print(self.total_order)
        print('\n')
        return self.total_order
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        print('Average sales amount per order')
        print('---------------------------------------------------------------------------\n')
        average_sales = self.total_value / self.total_order
        average_sales = round(average_sales, 2)
        print(average_sales)
        print('\n')
        return average_sales

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        print('How many different items are sold?')
        print('---------------------------------------------------------------------------\n')
        different_items = self.chipo['item_name'].unique()
        different_items = different_items.tolist()
        count_different_items = len(different_items)
        print(count_different_items)
        print('\n')
        return count_different_items

    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        print('Top x popular items bar plot')
        print('---------------------------------------------------------------------------\n')
        letter_items = letter_counter.items()
        letter_list = list(letter_items)
        letter_dataframe = pd.DataFrame(letter_list)
        sorted_letter_dataframe = letter_dataframe.sort_values(1, ascending=False).head(5)
        print(sorted_letter_dataframe)
        print('\n')
        result = sorted_letter_dataframe.plot.bar(x=0, y=1, title='Most popular items')
        plt.savefig('Bar_Plot.png',dpi=400)
        plt.show(block=True)
        
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        # 2. groupby the orders and sum it.
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        print('Number of items per order price scatter plot')
        print('---------------------------------------------------------------------------\n')
        orders_sum = self.chipo.groupby(['order_id']).sum()
        print(orders_sum)
        print('\n')
        orders_sum.plot.scatter(x='item_price',y='quantity', s=50, c='blue')
        plt.title("Number of items per order price")
        plt.xlabel("Order Price")
        plt.ylabel("Num Items")
        plt.savefig('Scatter_Plot.png',dpi=400)
        plt.show(block=True)
        pass
    
        

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    print('\n')
    assert count == 4622
    solution.info()
    count = solution.num_column()
    print(count)
    print('\n')
    assert count == 5
    solution.print_columns()
    item_name, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    #assert quantity == 159
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    
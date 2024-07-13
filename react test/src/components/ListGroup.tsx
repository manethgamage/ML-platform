import { Fragment } from "react/jsx-runtime";

function ListGroup(){
    let items = [
        'New york',
        'San Francisco',
        'tokyo'

    ];
    // items =[]

    // items.map(item=> <li>{item}</li>)


    return (
        <>
          <h1>List</h1>
          {items.length===0 && <p>No item Found</p>}
          <ul className="list-group">
            {items.map(item=> <li className="list-group-item" key={item} onClick={()=> console.log(item)}>{item}</li>)}
          </ul>
        </>
      );
}

export default ListGroup;